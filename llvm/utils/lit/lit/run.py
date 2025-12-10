import multiprocessing
import os
import platform
import time

import lit.Test
import lit.util
import lit.worker

# Windows has a limit of 60 workers per pool.
# This is defined in the multiprocessing module implementation.
# See: https://github.com/python/cpython/blob/6bc65c30ff1fd0b581a2c93416496fc720bc442c/Lib/concurrent/futures/process.py#L669-L672
WINDOWS_MAX_WORKERS_PER_POOL = 60


def _ceilDiv(a, b):
    return (a + b - 1) // b

class MaxFailuresError(Exception):
    pass


class TimeoutError(Exception):
    pass


class Run(object):
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

        # Create multiple pools
        pools = []
        for pool_size in workers_per_pool_list:
            pool = multiprocessing.Pool(
                pool_size, lit.worker.initialize, (self.lit_config, semaphores)
            )
            pools.append(pool)

        # Distribute tests across pools
        tests_per_pool = _ceilDiv(len(self.tests), num_pools)
        async_results = []

        for pool_idx, pool in enumerate(pools):
            start_idx = pool_idx * tests_per_pool
            end_idx = min(start_idx + tests_per_pool, len(self.tests))
            for test in self.tests[start_idx:end_idx]:
                ar = pool.apply_async(
                    lit.worker.execute, args=[test], callback=self.progress_callback
                )
                async_results.append(ar)

        # Close all pools
        for pool in pools:
            pool.close()

        try:
            self._wait_for(async_results, deadline)
        except:
            # Terminate all pools on exception
            for pool in pools:
                pool.terminate()
            raise
        finally:
            # Join all pools
            for pool in pools:
                pool.join()

    def _wait_for(self, async_results, deadline):
        timeout = deadline - time.time()
        for idx, ar in enumerate(async_results):
            try:
                test = ar.get(timeout)
            except multiprocessing.TimeoutError:
                raise TimeoutError()
            else:
                self._update_test(self.tests[idx], test)
                if test.isFailure():
                    self.failures += 1
                    if self.failures == self.max_failures:
                        raise MaxFailuresError()

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
