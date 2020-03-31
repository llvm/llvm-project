import multiprocessing
import os
import time

import lit.Test
import lit.util
import lit.worker


# No-operation semaphore for supporting `None` for parallelism_groups.
#   lit_config.parallelism_groups['my_group'] = None
class NopSemaphore(object):
    def acquire(self): pass
    def release(self): pass


class Run(object):
    """A concrete, configured testing run."""

    def __init__(self, tests, lit_config, workers, progress_callback,
                 max_failures, timeout):
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
        be given an UNRESOLVED result.
        """
        self.failure_count = 0
        self.hit_max_failures = False

        # Larger timeouts (one year, positive infinity) don't work on Windows.
        one_week = 7 * 24 * 60 * 60  # days * hours * minutes * seconds
        timeout = self.timeout or one_week
        deadline = time.time() + timeout

        self._execute(deadline)

        # Mark any tests that weren't run as UNRESOLVED.
        for test in self.tests:
            if test.result is None:
                test.setResult(lit.Test.Result(lit.Test.UNRESOLVED, '', 0.0))

    def _execute(self, deadline):
        semaphores = {
            k: NopSemaphore() if v is None else
            multiprocessing.BoundedSemaphore(v) for k, v in
            self.lit_config.parallelism_groups.items()}

        self._increase_process_limit()

        # Start a process pool. Copy over the data shared between all test runs.
        # FIXME: Find a way to capture the worker process stderr. If the user
        # interrupts the workers before we make it into our task callback, they
        # will each raise a KeyboardInterrupt exception and print to stderr at
        # the same time.
        pool = multiprocessing.Pool(self.workers, lit.worker.initialize,
                                    (self.lit_config, semaphores))

        self._install_win32_signal_handler(pool)

        async_results = [
            pool.apply_async(lit.worker.execute, args=[test],
                             callback=lambda t, i=idx: self._process_completed(t, i))
            for idx, test in enumerate(self.tests)]
        pool.close()

        for ar in async_results:
            timeout = deadline - time.time()
            try:
                ar.get(timeout)
            except multiprocessing.TimeoutError:
                # TODO(yln): print timeout error
                pool.terminate()
                break
            if self.hit_max_failures:
                pool.terminate()
                break
        pool.join()

    # TODO(yln): as the comment says.. this is racing with the main thread waiting
    # for results
    def _process_completed(self, test, idx):
        # Don't add any more test results after we've hit the maximum failure
        # count.  Otherwise we're racing with the main thread, which is going
        # to terminate the process pool soon.
        if self.hit_max_failures:
            return

        self.tests[idx] = test

        # Use test.isFailure() for correct XFAIL and XPASS handling
        if test.isFailure():
            self.failure_count += 1
            if self.failure_count == self.max_failures:
                self.hit_max_failures = True

        self.progress_callback(test)

    # TODO(yln): interferes with progress bar
    # Some tests use threads internally, and at least on Linux each of these
    # threads counts toward the current process limit. Try to raise the (soft)
    # process limit so that tests don't fail due to resource exhaustion.
    def _increase_process_limit(self):
        ncpus = lit.util.detectCPUs()
        desired_limit = self.workers * ncpus * 2 # the 2 is a safety factor

        # Importing the resource module will likely fail on Windows.
        try:
            import resource
            NPROC = resource.RLIMIT_NPROC

            soft_limit, hard_limit = resource.getrlimit(NPROC)
            desired_limit = min(desired_limit, hard_limit)

            if soft_limit < desired_limit:
                resource.setrlimit(NPROC, (desired_limit, hard_limit))
                self.lit_config.note('Raised process limit from %d to %d' % \
                                        (soft_limit, desired_limit))
        except Exception as ex:
            # Warn, unless this is Windows, in which case this is expected.
            if os.name != 'nt':
                self.lit_config.warning('Failed to raise process limit: %s' % ex)

    def _install_win32_signal_handler(self, pool):
        if lit.util.win32api is not None:
            def console_ctrl_handler(type):
                print('\nCtrl-C detected, terminating.')
                pool.terminate()
                pool.join()
                lit.util.abort_now()
                return True
            lit.util.win32api.SetConsoleCtrlHandler(console_ctrl_handler, True)
