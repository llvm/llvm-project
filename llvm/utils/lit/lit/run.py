import multiprocessing
import time

import lit.Test
import lit.util
import lit.worker

# No-operation semaphore for supporting `None` for parallelism_groups.
#   lit_config.parallelism_groups['my_group'] = None
class NopSemaphore(object):
    def acquire(self): pass
    def release(self): pass

def create_run(tests, lit_config, workers, progress_callback, timeout=None):
    # TODO(yln) assert workers > 0
    if workers == 1:
        return SerialRun(tests, lit_config, progress_callback, timeout)
    return ParallelRun(tests, lit_config, progress_callback, timeout, workers)

class Run(object):
    """A concrete, configured testing run."""

    def __init__(self, tests, lit_config, progress_callback, timeout):
        self.tests = tests
        self.lit_config = lit_config
        self.progress_callback = progress_callback
        self.timeout = timeout

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
        if not self.tests:
            return 0.0

        self.failure_count = 0
        self.hit_max_failures = False

        # Larger timeouts (one year, positive infinity) don't work on Windows.
        one_week = 7 * 24 * 60 * 60  # days * hours * minutes * seconds
        timeout = self.timeout or one_week

        start = time.time()
        deadline = start + timeout
        self._execute(deadline)
        end = time.time()

        # Mark any tests that weren't run as UNRESOLVED.
        for test in self.tests:
            if test.result is None:
                test.setResult(lit.Test.Result(lit.Test.UNRESOLVED, '', 0.0))

        return end - start

    def _process_result(self, test, result):
        # Don't add any more test results after we've hit the maximum failure
        # count.  Otherwise we're racing with the main thread, which is going
        # to terminate the process pool soon.
        if self.hit_max_failures:
            return

        # Update the parent process copy of the test. This includes the result,
        # XFAILS, REQUIRES, and UNSUPPORTED statuses.
        test.setResult(result)

        self.progress_callback(test)

        # If we've finished all the tests or too many tests have failed, notify
        # the main thread that we've stopped testing.
        self.failure_count += (result.code == lit.Test.FAIL)
        if self.lit_config.maxFailures and \
                self.failure_count == self.lit_config.maxFailures:
            self.hit_max_failures = True

class SerialRun(Run):
    def __init__(self, tests, lit_config, progress_callback, timeout):
        super(SerialRun, self).__init__(tests, lit_config, progress_callback, timeout)

    def _execute(self, deadline):
        # TODO(yln): ignores deadline
        for test in self.tests:
            result = lit.worker._execute(test, self.lit_config)
            self._process_result(test, result)
            if self.hit_max_failures:
                break

class ParallelRun(Run):
    def __init__(self, tests, lit_config, progress_callback, timeout, workers):
        super(ParallelRun, self).__init__(tests, lit_config, progress_callback, timeout)
        self.workers = workers

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

        # Install a console-control signal handler on Windows.
        if lit.util.win32api is not None:
            def console_ctrl_handler(type):
                print('\nCtrl-C detected, terminating.')
                pool.terminate()
                pool.join()
                lit.util.abort_now()
                return True
            lit.util.win32api.SetConsoleCtrlHandler(console_ctrl_handler, True)

        async_results = [
            pool.apply_async(lit.worker.execute, args=[test],
                callback=lambda r, t=test: self._process_result(t, r))
            for test in self.tests]
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
            self.lit_config.warning('Failed to raise process limit: %s' % ex)
