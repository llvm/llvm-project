import multiprocessing
import os
import platform
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
import concurrent.futures.process

import lit.Test
import lit.util
import lit.worker

# Windows has a limit of 60 workers per pool.
# This is defined in the multiprocessing module implementation.
# See: https://github.com/python/cpython/blob/6bc65c30ff1fd0b581a2c93416496fc720bc442c/Lib/concurrent/futures/process.py#L669-L672
WINDOWS_MAX_WORKERS_PER_POOL = 60

# Cap on outstanding futures, as a multiple of the worker count. Submitting
# every test up front deadlocks Python <= 3.11.5: each submit() writes one
# byte to executor's wakeup pipe while holding its shutdown lock, and the
# executor manager thread needs that same lock to drain the pipe. Once 1<<14
# undrained writes accumulate, submit() blocks holding the lock the manager
# needs (https://github.com/python/cpython/issues/105829). Bounding the
# outstanding futures bounds the undrained writes, so the pipe can never fill.
# The window must exceed workers + call-queue fetch (workers + 1) to keep
# every worker busy.
# TODO: Drop this workaround once lit's minimum Python version is >= 3.12
SUBMISSION_WINDOW_PER_WORKER = 4


def _ceilDiv(a, b):
    return (a + b - 1) // b

class MaxFailuresError(Exception):
    pass


class TimeoutError(Exception):
    pass


class WorkerCrashError(Exception):
    """A worker process died abrupty (segfault, OOM-kill, abort) instead of returning a result."""
    pass


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

    def _abort_executors(self, executors, future_to_test):
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
                for future in future_to_test:
                    future.cancel()
            # Killing worker processes can corrupt the executor's queues, which makes it
            # unsafe for its atexit hooks to join their threads. Disable those hooks
            # before terminating workers (a second ctrl-C should not bypass this cleanup).
            # This applies to call-queue feeder threads and management threads.
            # Otherwise, a thread blocked on a partially written pipe may require multiple
            # ctrl-C to unblock.
            # See: https://github.com/python/cpython/issues/125886
            # These threads are daemonic on Python 3.8, so disabling them is harmless.
            for ex in executors:
                if hasattr(ex, "_call_queue") and ex._call_queue is not None:
                    ex._call_queue.cancel_join_thread()
            if hasattr(concurrent.futures.process, "_threads_wakeups"):
                concurrent.futures.process._threads_wakeups.clear()
            tree_kill_ok, _ = lit.util.killProcessAndChildrenIsSupported()
            for ex in executors:
                for pid, proc in list((ex._processes or {}).items()):
                    if tree_kill_ok:
                        lit.util.killProcessAndChildren(pid)
                    else:
                        proc.kill()
            for ex in executors:
                for proc in list((ex._processes or {}).values()):
                    proc.join()  # reap: SIGKILL already delivered
            # TODO: Python>=3.14 adds ex.kill_workers(), which stops the workers cleanly
            # without corrupting the queues. However kill_workers() won't reap the
            # llc / FileCheck grandchildren the workers spawned.
            # https://github.com/python/cpython/issues/128041
        except Exception:
            pass

    def _execute(self, deadline):
        self._increase_process_limit()

        semaphores = {
            k: multiprocessing.BoundedSemaphore(v)
            for k, v in self.lit_config.parallelism_groups.items()
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

        num_pools = max(1, _ceilDiv(self.workers, max_workers_per_pool))

        # Distribute self.workers across num_pools as evenly as possible
        workers_per_pool_list = [self.workers // num_pools] * num_pools
        for pool_idx in range(self.workers % num_pools):
            workers_per_pool_list[pool_idx] += 1

        if num_pools > 1:
            self.lit_config.note(
                "Using %d pools balancing %d workers total distributed as %s (Windows worker limit workaround)"
                % (num_pools, self.workers, workers_per_pool_list)
            )

        executors = [
            ProcessPoolExecutor(
                max_workers=pool_size,
                initializer=lit.worker.initialize,
                initargs=(self.lit_config, semaphores),
            )
            for pool_size in workers_per_pool_list
        ]

        future_to_test = {}

        try:
            self._dispatch_and_wait(executors, future_to_test, deadline)
        except BaseException:
            self._abort_executors(executors, future_to_test)
            raise
        else:
            for ex in executors:
                # On macOS, Queue.join_thread() inside shutdown(wait=True)
                # deadlocks: join_executor_internals() calls it before
                # p.join(), but macOS requires the inverse order.
                # cancel_join_thread() makes join_thread() a no-op;
                # the feeder still delivers sentinels before the write end
                # closes.
                if hasattr(ex, "_call_queue") and ex._call_queue is not None:
                    ex._call_queue.cancel_join_thread()
                ex.shutdown(wait=True)

    def _dispatch_and_wait(self, executors, future_to_test, deadline):
        """Submits tests to executors and collects results as they complete.

        Bounds the number of futures outstanding at any time to at most
        window (see SUBMISSION_WINDOW_PER_WORKER), submitting exactly one
        new test for each one that completes. Submitting every test up
        front floods the executor's wakeup pipe and can deadlock submit()
        against the executor's manager thread on Python <= 3.11.5
        (https://github.com/python/cpython/issues/105829)

        Mutates future_to_test in place: adds an entry for every test
        submitted, and removes it once that test's result has been
        collected. On return, or if this call raises, future_to_test
        holds exactly the futures that have not yet been collected, which
        the caller's abort path relies on.

        Args:
            executors: The ProcessPoolExecutor pool(s) tests are dispatched to.
            future_to_test: A dict mapping each in-flight Future to its
              corresponding Test. Populated and drained by this call.
            deadline: The absolute time (as returned by time.time()) after
              which the call raises TimeoutError.

        Raises:
            TimeoutError: deadline passed with the tests still outstanding.
            MaxFailuresError: The number of failed tests reached self.max_failures.
            WorkerCrashError: A worker process died unexpectedly (e.g.
              segfault, OOM-kill) instead of returning a result.
        """
        try:
            window = int(
                os.getenv(
                    "LIT_SUBMISSION_WINDOW",
                    SUBMISSION_WINDOW_PER_WORKER * self.workers,
                )
            ) or len(self.tests)
            tests_iter = enumerate(self.tests)
            pending = set()
            future_to_index = {}

            def submit_next():
                """Submits the next not-yet-submitted test, if any.

                Returns:
                    True if a test was submitted, False if none remained.
                """
                for i, test in tests_iter:
                    ex = executors[i % len(executors)]
                    future = ex.submit(lit.worker.execute, test)
                    future_to_test[future] = test
                    future_to_index[future] = i
                    pending.add(future)
                    return True
                return False

            while len(pending) < window and submit_next():
                pass

            while pending:
                done, pending = wait(
                    pending,
                    timeout=deadline - time.time(),
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    raise TimeoutError()
                for future in sorted(done, key=lambda f: future_to_index[f]):
                    del future_to_index[future]
                    remote_test = future.result()
                    local_test = future_to_test.pop(future)
                    self._update_test(local_test, remote_test)
                    self.progress_callback(remote_test)
                    if remote_test.isFailure():
                        self.failures += 1
                        # max_failures is None or a positive int, never 0
                        # (cl_arguments.py's _positive_int enforces i > 0),
                        # so this equality check can't misfire on failures=0.
                        if self.failures == self.max_failures:
                            raise MaxFailuresError()
                    submit_next()

        except BrokenProcessPool as e:
            raise WorkerCrashError(str(e))

    # Update local test object "in place" from remote test object.  This
    # ensures that the original test object which is used for printing test
    # results reflects the changes.
    def _update_test(self, local_test, remote_test):
        # Needed for getMissingRequiredFeatures()
        local_test.requires = remote_test.requires
        local_test.result = remote_test.result

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
