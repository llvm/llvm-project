import abc
import multiprocessing
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures.process import BrokenProcessPool
import concurrent.futures.process

import lit.Test
import lit.util
import lit.worker

# Windows has a limit of 60 workers per pool.
# This is defined in the multiprocessing module implementation.
# See: https://github.com/python/cpython/blob/6bc65c30ff1fd0b581a2c93416496fc720bc442c/Lib/concurrent/futures/process.py#L669-L672
WINDOWS_MAX_WORKERS_PER_POOL = 60


def _ceilDiv(a, b):
    return (a + b - 1) // b


def _distribute_workers(workers, max_per_pool):
    """Split workers across as few pools as possible, each no longer than
    max_per_pool, balancing the counts as evenly as we can. POSIX always
    returns a single pool; the split only matters on Windows (60-worker cap) or
    when LIT_WINDOWS_MAX_WORKERS_PER_POOL forces it. Returns the list of
    per-pool worker counts."""
    num_pools = max(1, _ceilDiv(workers, max_per_pool))
    sizes = [workers // num_pools] * num_pools
    for i in range(workers % num_pools):
        sizes[i] += 1
    return sizes


class MaxFailuresError(Exception):
    pass


class TimeoutError(Exception):
    pass


class WorkerCrashError(Exception):
    """A worker process died abrupty (segfault, OOM-kill, abort) instead of returning a result."""
    pass


class ExecutionBackend(abc.ABC):
    """Manages the worker lifecycle and runs the entire test suite for a single Run.

    It operates at a higher level than a standard Executor. Instead of just running tasks
    and returning futures, a backend manages its own workers and handles its own abort
    logic (like instantly killing processes vs cleanly stopping threads). Heavy lifting
    like timeouts, failure limits, and saving results lives in Run._wait_for so all backends
    share it.

    Because of this, backends wrap an executor rather than subclassing it. Forcing
    concurrent.futures.Executor wouldn't work for an async engine (which uses an event loop,
    not futures) and lacks the abort() method that standard executors don't support.
    """

    __slots__ = ()

    @abc.abstractmethod
    def submit(self, tests):
        """Dispatch every test; return a {future: test} map for _wait_for."""
        raise NotImplementedError

    @abc.abstractmethod
    def abort(self):
        """Force stop all in-flight work (ctrl-C, --max-failures,
        --max-time, or a worker crash)."""
        raise NotImplementedError

    @abc.abstractmethod
    def shutdown(self):
        """Clean shutdown after every result has been collected."""
        raise NotImplementedError


class ProcessPoolBackend(ExecutionBackend):
    """Run each test in a standalone process using ProcessPoolExecutor.

    This is the default backend, moving the existing behavior here cleanly. It isolates
    everything process-specific away from Run. This includes handling process-limit
    bumps (RLIMIT_NPROC), using shared-memory semaphores for parallelism groups,
    splitting pools on Windows to respect the 60-worker limit, initializing each child
    process with lit_config, and handling hard SIGKILL teardowns.
    """

    def __init__(self, lit_config, workers):
        self.lit_config = lit_config
        self.workers = workers
        self.future_to_test = {}
        self._increase_process_limit()

        semaphores = {
            k: multiprocessing.BoundedSemaphore(v)
            for k, v in lit_config.parallelism_groups.items()
            if v is not None
        }
        # Windows has a limit of 60 workers per pool, so we need to use multiple pools
        # if we have more workers requested than the limit.
        # Also, allow to override the limit with the LIT_WINDOWS_MAX_WORKERS_PER_POOL environment variable.
        max_workers_per_pool = (
            WINDOWS_MAX_WORKERS_PER_POOL if os.name == "nt" else self.workers
        )
        max_workers_per_pool = int(
            os.getenv("LIT_WINDOWS_MAX_WORKERS_PER_POOL", max_workers_per_pool)
        )
        pool_sizes = _distribute_workers(workers, max_workers_per_pool)

        if len(pool_sizes) > 1:
            self.lit_config.note(
                "Using %d pools balancing %d workers total distributed as %s (Windows worker limit workaround)"
                % (len(pool_sizes), self.workers, pool_sizes)
            )

        self.executors = [
            ProcessPoolExecutor(
                max_workers=pool_size,
                initializer=lit.worker.initialize,
                initargs=(lit_config, semaphores),
            )
            for pool_size in pool_sizes
        ]

    def submit(self, tests):
        self.future_to_test = {}
        for i, test in enumerate(tests):
            ex = self.executors[i % len(self.executors)]
            self.future_to_test[ex.submit(lit.worker.execute, test)] = test
        return self.future_to_test

    def shutdown(self):
        for ex in self.executors:
            ex.shutdown(wait=True)

    def abort(self):
        """SIGKILL all workers on abort (ctrl-C, --max-failures, --max-time,
        worker crash). Pre-3.14 ProcessPoolExecutor has no force-stop."""
        try:
            # We don't call ex.shutdown() here: it joins the management thread,
            # which is blocked reading the queue we just corrupted.
            # On 3.8 / 3.9, cancel() races with the call-queue feeder thread and can
            # deadlock or corrupt the queue (https://github.com/python/cpython/issues/94440).
            # Skipping it is safe because we SIGKILL workers below, so no pending future
            # will ever be dispatched. cancel() on 3.10+ is a clean hint.
            if sys.version_info >= (3, 10):
                for future in self.future_to_test:
                    future.cancel()
            # Killing worker processes can corrupt the executor's queues, which makes it
            # unsafe for its atexit hooks to join their threads. Disable those hooks
            # before terminating workers (a second ctrl-C should not bypass this cleanup).
            # This applies to call-queue feeder threads and management threads.
            # Otherwise, a thread blocked on a partially written pipe may require multiple
            # ctrl-C to unblock.
            # See: https://github.com/python/cpython/issues/125886
            # These threads are daemonic on Python 3.8, so disabling them is harmless.
            for ex in self.executors:
                if hasattr(ex, "_call_queue") and ex._call_queue is not None:
                    ex._call_queue.cancel_join_thread()
            if hasattr(concurrent.futures.process, "_threads_wakeups"):
                concurrent.futures.process._threads_wakeups.clear()
            tree_kill_ok, _ = lit.util.killProcessAndChildrenIsSupported()
            for ex in self.executors:
                for pid, proc in list((ex._processes or {}).items()):
                    if tree_kill_ok:
                        lit.util.killProcessAndChildren(pid)
                    else:
                        proc.kill()
            # TODO: Python>=3.14 adds ex.kill_workers(), which stops the workers cleanly
            # without corrupting the queues. However kill_workers() won't reap the
            # llc / FileCheck grandchildren the workers spawned.
            # https://github.com/python/cpython/issues/128041
        except Exception:
            pass

    # TODO(yln): interferes with progress bar
    # Some tests use threads internally, and at least on Linux each of these
    # threads counts toward the current process limit. Try to raise the (soft)
    # process limit so that tests don't fail due to resource exhaustion.
    def _increase_process_limit(self):
        ncpus = lit.util.usable_core_count()
        desired_limit = self.workers * ncpus * 2  # the 2 is a safety factor

        # Importing the resource module will likely fail on Windows.
        try:
            import resource

            NPROC = resource.RLIMIT_NPROC

            soft_limit, hard_limit = resource.getrlimit(NPROC)
            desired_limit = min(desired_limit, hard_limit)

            if soft_limit < desired_limit:
                resource.setrlimit(NPROC, (desired_limit, hard_limit))
                self.lit_config.note(
                    "Raised process limit from %d to %d" % (soft_limit, desired_limit)
                )
        except Exception as ex:
            # Warn, unless this is Windows, z/OS, Solaris or Cygwin in which case this is expected.
            if (
                os.name != "nt"
                and platform.system() != "OS/390"
                and platform.system() != "SunOS"
                and platform.sys.platform != "cygwin"
            ):
                self.lit_config.warning("Failed to raise process limit: %s" % ex)


class Run:
    """A concrete, configured testing run."""

    def __init__(
        self, tests, lit_config, workers, progress_callback, max_failures, timeout
    ):
        self.tests = tests
        self.lit_config = lit_config
        self.workers = workers
        self.progress_callback = progress_callback
        self.max_failures = max_failures
        self.timeout = timeout
        assert workers > 0

    def execute(self):
        """
        Execute the tests in the run using up to the specified number of
        parallel tasks, and inform the caller of each individual result. The
        provided tests should be a subset of the tests available in this run
        object.

        The progress_callback will be invoked for each completed test.

        If timeout is non-None, it should be a time in seconds after which to
        stop executing tests.

        Returns the elapsed testing time.

        Upon completion, each test in the run will have its result
        computed. Tests which were not actually executed (for any reason) will
        be marked SKIPPED.
        """
        self.failures = 0

        # Larger timeouts (one year, positive infinity) don't work on Windows.
        one_week = 7 * 24 * 60 * 60  # days * hours * minutes * seconds
        timeout = self.timeout or one_week
        deadline = time.time() + timeout

        try:
            self._execute(deadline)
        finally:
            skipped = lit.Test.Result(lit.Test.SKIPPED)
            for test in self.tests:
                if test.result is None:
                    test.setResult(skipped)

    def _make_backend(self):
        """The single entry point for backend selection. Right now, it only supports
        the process pool. Future updates will add a ThreadPoolBackend behind
        --experimental-thread-workers, followed by an async engine.

        Without flags, it defaults to the process pool (current behavior).
        """
        return ProcessPoolBackend(self.lit_config, self.workers)

    def _execute(self, deadline):
        backend = self._make_backend()
        future_to_test = backend.submit(self.tests)
        try:
            self._wait_for(future_to_test, deadline)
        except BaseException:
            backend.abort()
            raise
        else:
            backend.shutdown()

    def _wait_for(self, future_to_test, deadline):
        try:
            for future in as_completed(future_to_test, timeout=deadline - time.time()):
                remote_test = future.result()
                local_test = future_to_test[future]
                self._update_test(local_test, remote_test)
                self.progress_callback(remote_test)
                if remote_test.isFailure():
                    self.failures += 1
                    if self.failures == self.max_failures:
                        raise MaxFailuresError()
        except FuturesTimeoutError:
            raise TimeoutError()
        except BrokenProcessPool as e:
            raise WorkerCrashError(str(e))

    # Update local test object "in place" from remote test object.  This
    # ensures that the original test object which is used for printing test
    # results reflects the changes.
    def _update_test(self, local_test, remote_test):
        # Needed for getMissingRequiredFeatures()
        local_test.requires = remote_test.requires
        local_test.result = remote_test.result

