"""
The functions in this module are meant to run on a separate worker process.
Exception: in single process mode _execute is called directly.

For efficiency, we copy all data needed to execute all tests into each worker
and store it in global variables. This reduces the cost of each task.
"""
import contextlib
import os
import random
import signal
import time
import traceback

import lit.Test
import lit.util
from lit.TestRunner import TestUpdaterException


_lit_config = None
_parallelism_semaphores = None
_load_limit = None
# Shared multiprocessing.Array('d', [last_tick, running, ceiling]).
_load_state = None
# Hard cap on the dynamic ceiling -- the -j value.
_max_ceiling = 1

# Indices into _load_state.  Must match the layout allocated in run.py.
_S_LAST_TICK = 0
_S_RUNNING = 1
_S_CEILING = 2

# Tunables for the load-average gate.  Exposed via environment variables
# so users can experiment without having to recompile or edit the source.
_LOAD_POLL_INTERVAL = float(os.environ.get("LIT_LOAD_POLL", "1.0"))
_LOAD_TICK_INTERVAL = float(os.environ.get("LIT_LOAD_TICK", "1.0"))
# Maximum +/- adjustment of the (float) ceiling per controller tick when the
# system is saturated (load <= 0 or load >= 2*limit).  Near the limit the
# adjustment shrinks proportionally, damping oscillations.
_LOAD_MAX_STEP = float(os.environ.get("LIT_LOAD_STEP", "1.0"))

# Whether os.getloadavg() is available on this platform.  Cached because
# the gate consults it on every poll iteration.
_HAS_LOADAVG = hasattr(os, "getloadavg")

def initialize(
    lit_config,
    parallelism_semaphores,
    workers_max,
    workers_min=1,
    load_limit=None,
    load_state=None,
):
    """Copy data shared by all test executions into worker processes"""
    global _lit_config
    global _parallelism_semaphores
    global _load_limit
    global _load_state
    global _min_ceiling
    global _max_ceiling
    _lit_config = lit_config
    _parallelism_semaphores = parallelism_semaphores
    _load_limit = load_limit
    _load_state = load_state
    _min_ceiling = max(1, int(workers_min))
    _max_ceiling = max(1, int(workers_max))

    # We use the following strategy for dealing with Ctrl+C/KeyboardInterrupt in
    # subprocesses created by the multiprocessing.Pool.
    # https://noswap.com/blog/python-multiprocessing-keyboardinterrupt
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _tick_ceiling(now, load1m):
    """Adjust the dynamic concurrency ceiling (proportional controller).

    Must be called with _load_state's lock held.  Performs at most one
    adjustment per LIT_LOAD_TICK seconds.  The adjustment is
    proportional to the normalized signed distance between current load
    and --load-limit, clamped to a maximum of +/- LIT_LOAD_STEP per
    tick:

        distance = clamp((load - limit) / limit, -1, +1)
        delta    = -distance * LIT_LOAD_STEP
        ceiling  = clamp(ceiling + delta, 1.0, workers)

    Effect:
      * load == limit       -> delta == 0,        ceiling holds steady
      * load == 0           -> delta == +STEP,    fastest ramp-up
      * load == 2 * limit   -> delta == -STEP,    fastest shrink
      * load just above/below limit -> tiny delta, no oscillation

    The ceiling is stored as a float; the gate compares `running` to
    int(ceiling), so concurrency only changes once the float crosses an
    integer boundary -- this naturally damps churn near the limit.
    """
    if (now - _load_state[_S_LAST_TICK]) < _LOAD_TICK_INTERVAL:
        return
    _load_state[_S_LAST_TICK] = now

    distance = max(-1.0, min(1.0, (load1m - _load_limit) / _load_limit))
    new_ceiling = _load_state[_S_CEILING] - distance * _LOAD_MAX_STEP
    _load_state[_S_CEILING] = max(float(_min_ceiling), min(float(_max_ceiling), new_ceiling))


def _acquire_run_slot():
    """Block until it is OK to start a new test.

    With --load-limit set, the number of concurrent tests is governed by
    a dynamic ceiling whose movement is proportional to the (signed)
    distance between the 1-minute system load and --load-limit; see
    _tick_ceiling().  A new test is admitted when:
      * running == 0             (forward progress - never wait forever), or
      * running <  int(ceiling)  (within the current concurrency budget), or
      * running <  _min_ceiling  (optional).

    The ceiling is stored as a float internally so that near-limit
    fluctuations of well under one worker accumulate smoothly instead of
    flipping concurrency on every tick.
    """
    if _load_state is None or not _HAS_LOADAVG:
        return

    while True:
        load1m = os.getloadavg()[0]
        now = time.time()

        with _load_state.get_lock():
            _tick_ceiling(now, load1m)

            running = int(_load_state[_S_RUNNING])
            ceiling_f = float(_load_state[_S_CEILING])
            ceiling = int(ceiling_f)

            if running == 0 or running < _min_ceiling or running < ceiling:
                running += 1
                _load_state[_S_RUNNING] = running
                break

        # Jittered sleep so all idle workers don't re-check in lockstep.
        time.sleep(_LOAD_POLL_INTERVAL + random.uniform(0.0, _LOAD_POLL_INTERVAL))

def _release_run_slot():
    if _load_state is None or not _HAS_LOADAVG:
        return

    with _load_state.get_lock():
        if _load_state[_S_RUNNING] > 0:
            _load_state[_S_RUNNING] -= 1

def execute(test):
    """Run one test in a multiprocessing.Pool

    Side effects in this function and functions it calls are not visible in the
    main lit process.

    Arguments and results of this function are pickled, so they should be cheap
    to copy.
    """
    if _load_limit:
        _acquire_run_slot()
    try:
        with _get_parallelism_semaphore(test):
            result = _execute(test, _lit_config)
    finally:
        if _load_limit:
            _release_run_slot()

    test.setResult(result)
    return test


# TODO(python3): replace with contextlib.nullcontext
@contextlib.contextmanager
def NopSemaphore():
    yield


def _get_parallelism_semaphore(test):
    pg = test.config.parallelism_group
    if callable(pg):
        pg = pg(test)
    return _parallelism_semaphores.get(pg, NopSemaphore())


# Do not inline! Directly used by LitTestCase.py
def _execute(test, lit_config):
    start = time.time()
    result = _execute_test_handle_errors(test, lit_config)
    result.elapsed = time.time() - start
    result.start = start
    result.pid = os.getpid()
    return result


def _execute_test_handle_errors(test, lit_config):
    try:
        result = test.config.test_format.execute(test, lit_config)
        return _adapt_result(result)
    except TestUpdaterException as e:
        if lit_config.debug:
            raise
        return lit.Test.Result(lit.Test.UNRESOLVED, str(e))
    except:
        if lit_config.debug:
            raise
        output = "Exception during script execution:\n"
        output += traceback.format_exc()
        output += "\n"
        return lit.Test.Result(lit.Test.UNRESOLVED, output)


# Support deprecated result from execute() which returned the result
# code and additional output as a tuple.
def _adapt_result(result):
    if isinstance(result, lit.Test.Result):
        return result
    assert isinstance(result, tuple)
    code, output = result
    return lit.Test.Result(code, output)
