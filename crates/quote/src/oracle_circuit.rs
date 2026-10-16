//! Per-endpoint circuit breaker for oracle HTTP requests.
//!
//! Orders bind their oracle URL when they are deployed, so an order whose
//! oracle endpoint was retired keeps naming it until the order is removed.
//! Without a breaker every quote refetches that endpoint and gets the same
//! 404. The breaker opens after repeated endpoint-level HTTP 4xx responses,
//! skips the endpoint while open, and lets one probe through after a cooldown
//! in case the endpoint came back.
//!
//! Only responses that say the endpoint itself is missing count as failures.
//! Timeouts, connection errors and 5xx never count: oracles shut down off
//! hours and refuse quotes with 503 (`no_live_quote`, `expired_quote`, ...),
//! and those endpoints must keep being asked.
//!
//! The breaker is disabled on WASM, where `std::time::Instant` is unavailable.
#![cfg_attr(target_family = "wasm", allow(dead_code))]

use reqwest::StatusCode;
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tracing::{debug, info, warn};

/// Endpoint-level HTTP 4xx responses within [`ORACLE_CIRCUIT_FAILURE_WINDOW`]
/// that open an endpoint's circuit.
pub const ORACLE_CIRCUIT_FAILURE_THRESHOLD: usize = 3;
/// Sliding window in which [`ORACLE_CIRCUIT_FAILURE_THRESHOLD`] failures open
/// the circuit.
pub const ORACLE_CIRCUIT_FAILURE_WINDOW: Duration = Duration::from_secs(10 * 60);
/// How long an open circuit skips its endpoint before one probe is let through.
pub const ORACLE_CIRCUIT_COOLDOWN: Duration = Duration::from_secs(15 * 60);

/// What one completed oracle request says about its endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CircuitOutcome {
    /// Any 2xx response: the endpoint exists and answers.
    Success,
    /// An HTTP status saying the endpoint itself is missing.
    EndpointFailure,
    /// Says nothing about whether the endpoint exists: timeouts, connection
    /// errors, 5xx (including the oracle's off-hours 503 `no_live_quote`),
    /// rate limits, and 4xx statuses that depend on the request body.
    Inconclusive,
}

/// Classify an HTTP status for the breaker.
///
/// 404 Not Found, 405 Method Not Allowed and 410 Gone describe the endpoint,
/// whatever was posted to it. 400 Bad Request counts only for
/// `allowFailure=true` batches: the oracle reports per-item refusals (bad
/// pair, unknown token, invalid IO index) as `error` items in a 200 response
/// there, so a whole-request 400 means the endpoint does not speak the batch
/// schema. A strict request's 400 can be one order's refusal, and the URL may
/// be shared with healthy orders, so it never counts.
pub(crate) fn classify_status(status: StatusCode, allow_failure_batch: bool) -> CircuitOutcome {
    if status.is_success() {
        return CircuitOutcome::Success;
    }
    match status {
        StatusCode::NOT_FOUND | StatusCode::METHOD_NOT_ALLOWED | StatusCode::GONE => {
            CircuitOutcome::EndpointFailure
        }
        StatusCode::BAD_REQUEST if allow_failure_batch => CircuitOutcome::EndpointFailure,
        _ => CircuitOutcome::Inconclusive,
    }
}

/// Monotonic clock, injectable so tests never sleep.
pub(crate) trait CircuitClock: Send + Sync {
    fn now(&self) -> Instant;
}

pub(crate) struct SystemClock;

impl CircuitClock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

#[derive(Debug)]
enum CircuitState {
    Closed,
    Open { until: Instant },
    HalfOpen { probe_in_flight: bool },
}

#[derive(Debug)]
struct CircuitEntry {
    state: CircuitState,
    /// Endpoint failures inside the window while closed (at most threshold).
    recent_failures: VecDeque<Instant>,
    /// Calls skipped since the circuit last opened, reported when it closes.
    skipped_since_open: u64,
}

impl CircuitEntry {
    fn new() -> Self {
        Self {
            state: CircuitState::Closed,
            recent_failures: VecDeque::new(),
            skipped_since_open: 0,
        }
    }
}

/// Why a request was refused without being sent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CircuitRejection {
    /// Time until the next probe may go out; zero while a probe is in flight.
    pub retry_in: Duration,
}

/// Shared per-endpoint breaker state.
pub(crate) struct OracleCircuitBreaker {
    clock: Arc<dyn CircuitClock>,
    entries: Mutex<HashMap<String, CircuitEntry>>,
}

impl OracleCircuitBreaker {
    pub(crate) fn new(clock: Arc<dyn CircuitClock>) -> Self {
        Self {
            clock,
            entries: Mutex::new(HashMap::new()),
        }
    }

    fn entries(&self) -> std::sync::MutexGuard<'_, HashMap<String, CircuitEntry>> {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Ask to send one request to `endpoint`.
    ///
    /// Closed circuits admit every request. An open circuit refuses until its
    /// cooldown ends, then admits exactly one probe; other callers are refused
    /// until the probe is recorded or dropped.
    pub(crate) fn admit(&self, endpoint: &str) -> Result<CircuitPermit<'_>, CircuitRejection> {
        let now = self.clock.now();
        let mut entries = self.entries();
        let Some(entry) = entries.get_mut(endpoint) else {
            return Ok(self.permit(endpoint, false));
        };
        match entry.state {
            CircuitState::Closed => Ok(self.permit(endpoint, false)),
            CircuitState::Open { until } if now >= until => {
                entry.state = CircuitState::HalfOpen {
                    probe_in_flight: true,
                };
                Ok(self.permit(endpoint, true))
            }
            CircuitState::HalfOpen {
                probe_in_flight: false,
            } => {
                entry.state = CircuitState::HalfOpen {
                    probe_in_flight: true,
                };
                Ok(self.permit(endpoint, true))
            }
            CircuitState::Open { until } => {
                entry.skipped_since_open += 1;
                Err(CircuitRejection {
                    retry_in: until.saturating_duration_since(now),
                })
            }
            CircuitState::HalfOpen {
                probe_in_flight: true,
            } => {
                entry.skipped_since_open += 1;
                Err(CircuitRejection {
                    retry_in: Duration::ZERO,
                })
            }
        }
    }

    fn permit(&self, endpoint: &str, probe: bool) -> CircuitPermit<'_> {
        CircuitPermit {
            breaker: self,
            endpoint: endpoint.to_owned(),
            probe,
            recorded: false,
        }
    }

    fn record(&self, endpoint: &str, probe: bool, outcome: CircuitOutcome) {
        let now = self.clock.now();
        let mut entries = self.entries();
        match outcome {
            CircuitOutcome::Success => {
                let Some(entry) = entries.remove(endpoint) else {
                    return;
                };
                if !matches!(entry.state, CircuitState::Closed) {
                    info!(
                        endpoint,
                        skipped_count = entry.skipped_since_open,
                        "oracle circuit closed: endpoint answered again"
                    );
                }
            }
            CircuitOutcome::EndpointFailure => {
                let entry = entries
                    .entry(endpoint.to_owned())
                    .or_insert_with(CircuitEntry::new);
                if probe {
                    entry.state = CircuitState::Open {
                        until: now + ORACLE_CIRCUIT_COOLDOWN,
                    };
                    debug!(endpoint, "oracle circuit probe failed; circuit stays open");
                    return;
                }
                if !matches!(entry.state, CircuitState::Closed) {
                    // A request admitted before the circuit opened.
                    return;
                }
                entry.recent_failures.push_back(now);
                while entry.recent_failures.front().is_some_and(|failed_at| {
                    now.saturating_duration_since(*failed_at) >= ORACLE_CIRCUIT_FAILURE_WINDOW
                }) {
                    entry.recent_failures.pop_front();
                }
                if entry.recent_failures.len() >= ORACLE_CIRCUIT_FAILURE_THRESHOLD {
                    let failure_count = entry.recent_failures.len();
                    entry.recent_failures.clear();
                    entry.skipped_since_open = 0;
                    entry.state = CircuitState::Open {
                        until: now + ORACLE_CIRCUIT_COOLDOWN,
                    };
                    warn!(
                        endpoint,
                        failure_count,
                        window_secs = ORACLE_CIRCUIT_FAILURE_WINDOW.as_secs(),
                        cooldown_secs = ORACLE_CIRCUIT_COOLDOWN.as_secs(),
                        "oracle circuit opened: endpoint answered HTTP 4xx repeatedly; skipping it"
                    );
                }
            }
            CircuitOutcome::Inconclusive => {
                if !probe {
                    return;
                }
                // The probe could not tell whether the endpoint is back (the
                // oracle may be shut off hours); wait another cooldown.
                if let Some(entry) = entries.get_mut(endpoint) {
                    entry.state = CircuitState::Open {
                        until: now + ORACLE_CIRCUIT_COOLDOWN,
                    };
                }
                debug!(
                    endpoint,
                    "oracle circuit probe inconclusive; circuit stays open"
                );
            }
        }
    }

    /// A probe dropped before it finished (cancelled caller): let the next
    /// caller probe instead.
    fn release_probe(&self, endpoint: &str) {
        let mut entries = self.entries();
        if let Some(entry) = entries.get_mut(endpoint) {
            if matches!(
                entry.state,
                CircuitState::HalfOpen {
                    probe_in_flight: true
                }
            ) {
                entry.state = CircuitState::HalfOpen {
                    probe_in_flight: false,
                };
            }
        }
    }
}

/// Admission to send one request. Record the outcome with
/// [`CircuitPermit::record`]; dropping it unrecorded frees a probe slot.
pub(crate) struct CircuitPermit<'a> {
    breaker: &'a OracleCircuitBreaker,
    endpoint: String,
    probe: bool,
    recorded: bool,
}

impl CircuitPermit<'_> {
    pub(crate) fn record(mut self, outcome: CircuitOutcome) {
        self.recorded = true;
        self.breaker.record(&self.endpoint, self.probe, outcome);
    }

    #[cfg(all(test, not(target_family = "wasm")))]
    pub(crate) fn is_probe(&self) -> bool {
        self.probe
    }
}

impl Drop for CircuitPermit<'_> {
    fn drop(&mut self) {
        if !self.recorded && self.probe {
            self.breaker.release_probe(&self.endpoint);
        }
    }
}

// Tests use `Instant`, which is unavailable on WASM.
#[cfg(all(test, not(target_family = "wasm")))]
pub(crate) mod tests {
    use super::*;

    /// Manually advanced clock.
    pub(crate) struct FakeClock {
        now: Mutex<Instant>,
    }

    impl FakeClock {
        pub(crate) fn new() -> Arc<Self> {
            Arc::new(Self {
                now: Mutex::new(Instant::now()),
            })
        }

        pub(crate) fn advance(&self, by: Duration) {
            *self.now.lock().unwrap() += by;
        }
    }

    impl CircuitClock for FakeClock {
        fn now(&self) -> Instant {
            *self.now.lock().unwrap()
        }
    }

    const DEAD: &str = "https://st0x-oracle-server.fly.dev/context";
    const LIVE: &str = "https://oracle.t0trade.com/context/v7";

    fn breaker() -> (Arc<FakeClock>, OracleCircuitBreaker) {
        let clock = FakeClock::new();
        (clock.clone(), OracleCircuitBreaker::new(clock))
    }

    fn send(breaker: &OracleCircuitBreaker, endpoint: &str, outcome: CircuitOutcome) {
        breaker
            .admit(endpoint)
            .expect("request should be admitted")
            .record(outcome);
    }

    fn fail_until_open(breaker: &OracleCircuitBreaker, endpoint: &str) {
        for _ in 0..ORACLE_CIRCUIT_FAILURE_THRESHOLD {
            send(breaker, endpoint, CircuitOutcome::EndpointFailure);
        }
        assert!(breaker.admit(endpoint).is_err(), "circuit should be open");
    }

    #[test]
    fn test_classify_status_counts_only_endpoint_level_4xx() {
        for status in [
            StatusCode::NOT_FOUND,
            StatusCode::METHOD_NOT_ALLOWED,
            StatusCode::GONE,
        ] {
            assert_eq!(
                classify_status(status, false),
                CircuitOutcome::EndpointFailure
            );
            assert_eq!(
                classify_status(status, true),
                CircuitOutcome::EndpointFailure
            );
        }
        assert_eq!(
            classify_status(StatusCode::BAD_REQUEST, true),
            CircuitOutcome::EndpointFailure
        );
        // A strict request's 400 can be one order's refusal on a shared URL.
        assert_eq!(
            classify_status(StatusCode::BAD_REQUEST, false),
            CircuitOutcome::Inconclusive
        );
        // 503 is how the oracle answers no_live_quote/expired_quote off hours.
        for status in [
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
            StatusCode::GATEWAY_TIMEOUT,
            StatusCode::REQUEST_TIMEOUT,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
        ] {
            assert_eq!(classify_status(status, true), CircuitOutcome::Inconclusive);
            assert_eq!(classify_status(status, false), CircuitOutcome::Inconclusive);
        }
        assert_eq!(
            classify_status(StatusCode::OK, false),
            CircuitOutcome::Success
        );
    }

    #[test]
    fn test_three_endpoint_failures_open_the_circuit() {
        let (_, breaker) = breaker();
        send(&breaker, DEAD, CircuitOutcome::EndpointFailure);
        send(&breaker, DEAD, CircuitOutcome::EndpointFailure);
        assert!(breaker.admit(DEAD).is_ok(), "two failures must not open");
        send(&breaker, DEAD, CircuitOutcome::EndpointFailure);

        let rejection = breaker.admit(DEAD).err().expect("circuit should be open");
        assert_eq!(rejection.retry_in, ORACLE_CIRCUIT_COOLDOWN);
    }

    #[test]
    fn test_inconclusive_outcomes_never_open_the_circuit() {
        let (clock, breaker) = breaker();
        for _ in 0..100 {
            send(&breaker, LIVE, CircuitOutcome::Inconclusive);
            clock.advance(Duration::from_secs(1));
        }
        assert!(breaker.admit(LIVE).is_ok());
    }

    #[test]
    fn test_inconclusive_outcomes_do_not_reset_or_add_to_the_count() {
        let (_, breaker) = breaker();
        send(&breaker, DEAD, CircuitOutcome::EndpointFailure);
        send(&breaker, DEAD, CircuitOutcome::Inconclusive);
        send(&breaker, DEAD, CircuitOutcome::EndpointFailure);
        send(&breaker, DEAD, CircuitOutcome::Inconclusive);
        assert!(breaker.admit(DEAD).is_ok());
        send(&breaker, DEAD, CircuitOutcome::EndpointFailure);
        assert!(breaker.admit(DEAD).is_err());
    }

    #[test]
    fn test_success_resets_the_failure_count() {
        let (_, breaker) = breaker();
        send(&breaker, DEAD, CircuitOutcome::EndpointFailure);
        send(&breaker, DEAD, CircuitOutcome::EndpointFailure);
        send(&breaker, DEAD, CircuitOutcome::Success);
        send(&breaker, DEAD, CircuitOutcome::EndpointFailure);
        send(&breaker, DEAD, CircuitOutcome::EndpointFailure);
        assert!(breaker.admit(DEAD).is_ok());
    }

    #[test]
    fn test_failures_outside_the_window_do_not_count() {
        let (clock, breaker) = breaker();
        send(&breaker, DEAD, CircuitOutcome::EndpointFailure);
        send(&breaker, DEAD, CircuitOutcome::EndpointFailure);
        clock.advance(ORACLE_CIRCUIT_FAILURE_WINDOW);
        send(&breaker, DEAD, CircuitOutcome::EndpointFailure);
        assert!(breaker.admit(DEAD).is_ok());
        send(&breaker, DEAD, CircuitOutcome::EndpointFailure);
        send(&breaker, DEAD, CircuitOutcome::EndpointFailure);
        assert!(breaker.admit(DEAD).is_err());
    }

    #[test]
    fn test_open_circuit_rejects_until_cooldown_then_admits_one_probe() {
        let (clock, breaker) = breaker();
        fail_until_open(&breaker, DEAD);

        clock.advance(ORACLE_CIRCUIT_COOLDOWN - Duration::from_secs(1));
        let rejection = breaker.admit(DEAD).err().expect("still open");
        assert_eq!(rejection.retry_in, Duration::from_secs(1));

        clock.advance(Duration::from_secs(1));
        let probe = breaker.admit(DEAD).expect("cooldown over: one probe");
        assert!(probe.is_probe());
        for _ in 0..5 {
            assert_eq!(
                breaker.admit(DEAD).err(),
                Some(CircuitRejection {
                    retry_in: Duration::ZERO
                }),
                "only one probe at a time"
            );
        }
        drop(probe);
    }

    #[test]
    fn test_probe_failure_reopens_for_another_cooldown() {
        let (clock, breaker) = breaker();
        fail_until_open(&breaker, DEAD);
        clock.advance(ORACLE_CIRCUIT_COOLDOWN);

        breaker
            .admit(DEAD)
            .expect("probe")
            .record(CircuitOutcome::EndpointFailure);

        let rejection = breaker.admit(DEAD).err().expect("reopened");
        assert_eq!(rejection.retry_in, ORACLE_CIRCUIT_COOLDOWN);
        clock.advance(ORACLE_CIRCUIT_COOLDOWN);
        assert!(breaker.admit(DEAD).expect("next probe").is_probe());
    }

    #[test]
    fn test_inconclusive_probe_keeps_the_circuit_open() {
        let (clock, breaker) = breaker();
        fail_until_open(&breaker, DEAD);
        clock.advance(ORACLE_CIRCUIT_COOLDOWN);

        breaker
            .admit(DEAD)
            .expect("probe")
            .record(CircuitOutcome::Inconclusive);

        let rejection = breaker.admit(DEAD).err().expect("still open");
        assert_eq!(rejection.retry_in, ORACLE_CIRCUIT_COOLDOWN);
    }

    #[test]
    fn test_probe_success_closes_the_circuit() {
        let (clock, breaker) = breaker();
        fail_until_open(&breaker, DEAD);
        clock.advance(ORACLE_CIRCUIT_COOLDOWN);

        breaker
            .admit(DEAD)
            .expect("probe")
            .record(CircuitOutcome::Success);

        for _ in 0..5 {
            let permit = breaker.admit(DEAD).expect("closed");
            assert!(!permit.is_probe());
            permit.record(CircuitOutcome::Success);
        }
        // Closed with a fresh count: two failures do not reopen it.
        send(&breaker, DEAD, CircuitOutcome::EndpointFailure);
        send(&breaker, DEAD, CircuitOutcome::EndpointFailure);
        assert!(breaker.admit(DEAD).is_ok());
    }

    #[test]
    fn test_dropped_probe_lets_the_next_caller_probe() {
        let (clock, breaker) = breaker();
        fail_until_open(&breaker, DEAD);
        clock.advance(ORACLE_CIRCUIT_COOLDOWN);

        let probe = breaker.admit(DEAD).expect("probe");
        assert!(breaker.admit(DEAD).is_err());
        drop(probe);

        assert!(breaker.admit(DEAD).expect("new probe").is_probe());
    }

    #[test]
    fn test_late_failures_from_requests_admitted_before_opening_are_ignored() {
        let (clock, breaker) = breaker();
        let in_flight = breaker.admit(DEAD).expect("admitted while closed");
        fail_until_open(&breaker, DEAD);
        clock.advance(ORACLE_CIRCUIT_COOLDOWN - Duration::from_secs(60));
        in_flight.record(CircuitOutcome::EndpointFailure);

        let rejection = breaker.admit(DEAD).err().expect("still open");
        assert_eq!(rejection.retry_in, Duration::from_secs(60));
    }

    #[test]
    fn test_endpoints_are_independent() {
        let (_, breaker) = breaker();
        fail_until_open(&breaker, DEAD);
        assert!(breaker.admit(LIVE).is_ok());
        send(&breaker, LIVE, CircuitOutcome::EndpointFailure);
        send(&breaker, LIVE, CircuitOutcome::EndpointFailure);
        assert!(breaker.admit(LIVE).is_ok());
        assert!(breaker.admit(DEAD).is_err());
    }
}
