import multiprocessing
import os
import time

import lit.Test
import lit.util
import lit.worker


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
        print("Endill 36500")
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
            print("Endill 36510")
            self._execute(deadline)
            print("Endill 36520")
        finally:
            print("Endill 36530")
            skipped = lit.Test.Result(lit.Test.SKIPPED)
            print("Endill 36540")
            for test in self.tests:
                if test.result is None:
                    test.setResult(skipped)
            print("Endill 36560")
            

    def _execute(self, deadline):
        print("Endill 365100")
        self._increase_process_limit()
        print("Endill 365110")

        semaphores = {
            k: multiprocessing.BoundedSemaphore(v)
            for k, v in self.lit_config.parallelism_groups.items()
            if v is not None
        }
        print("Endill 365120")

        pool = multiprocessing.Pool(
            self.workers, lit.worker.initialize, (self.lit_config, semaphores)
        )
        print("Endill 365130")

        async_results = [
            pool.apply_async(
                lit.worker.execute, args=[test], callback=self.progress_callback
            )
            for test in self.tests
        ]
        print("Endill 365140")
        pool.close()
        print("Endill 365150")

        try:
            print("Endill 365160")
            self._wait_for(async_results, deadline)
            print("Endill 365170")
        except:
            print("Endill 365180")
            pool.terminate()
            print("Endill 365190")
            raise
        finally:
            print("Endill 3651100")
            pool.join()
            print("Endill 3651110")

    def _wait_for(self, async_results, deadline):
        print("Endill 3651600")
        timeout = deadline - time.time()
        for idx, ar in enumerate(async_results):
            print("Endill 3651610")
            try:
                print("Endill 3651620")
                test = ar.get(timeout)
                print("Endill 3651630")
                print("Endill: test.file_path: {}".format(test.file_path))
                print("Endill: test.path_in_suite: {}".format(test.path_in_suite))
                print("Endill 3651635")
            except multiprocessing.TimeoutError:
                print("Endill 3651640")
                raise TimeoutError()
            else:
                print("Endill 3651650")
                self._update_test(self.tests[idx], test)
                print("Endill 3651660")
                if test.isFailure():
                    self.failures += 1
                    if self.failures == self.max_failures:
                        raise MaxFailuresError()
        print("Endill 3651670") 

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
            # Warn, unless this is Windows, in which case this is expected.
            if os.name != "nt":
                self.lit_config.warning("Failed to raise process limit: %s" % ex)
