import multiprocessing
import time

import lit.Test
import lit.util
import lit.worker

class _Display(object):
    def __init__(self, display, provider, maxFailures):
        self.display = display
        self.provider = provider
        self.maxFailures = maxFailures or object()
        self.failedCount = 0
    def update(self, test):
        self.display.update(test)
        self.failedCount += (test.result.code == lit.Test.FAIL)
        if self.failedCount == self.maxFailures:
            self.provider.cancel()

class Run(object):
    """
    This class represents a concrete, configured testing run.
    """

    def __init__(self, lit_config, tests):
        self.lit_config = lit_config
        self.tests = tests
        # Set up semaphores to limit parallelism of certain classes of tests.
        # For example, some ASan tests require lots of virtual memory and run
        # faster with less parallelism on OS X.
        self.parallelism_semaphores = \
                {k: multiprocessing.BoundedSemaphore(v) for k, v in
                 self.lit_config.parallelism_groups.items()}

    def execute_tests_in_pool(self, jobs, max_time):
        # We need to issue many wait calls, so compute the final deadline and
        # subtract time.time() from that as we go along.
        deadline = None
        if max_time:
            deadline = time.time() + max_time

        # Start a process pool. Copy over the data shared between all test runs.
        # FIXME: Find a way to capture the worker process stderr. If the user
        # interrupts the workers before we make it into our task callback, they
        # will each raise a KeyboardInterrupt exception and print to stderr at
        # the same time.
        pool = multiprocessing.Pool(jobs, lit.worker.initializer,
                                    (self.lit_config,
                                     self.parallelism_semaphores))

        # Install a console-control signal handler on Windows.
        if lit.util.win32api is not None:
            def console_ctrl_handler(type):
                print('\nCtrl-C detected, terminating.')
                pool.terminate()
                pool.join()
                lit.util.abort_now()
                return True
            lit.util.win32api.SetConsoleCtrlHandler(console_ctrl_handler, True)

        try:
            async_results = [pool.apply_async(lit.worker.run_one_test,
                                              args=(test_index, test),
                                              callback=self.consume_test_result)
                             for test_index, test in enumerate(self.tests)]
            pool.close()

            # Wait for all results to come in. The callback that runs in the
            # parent process will update the display.
            for a in async_results:
                if deadline:
                    a.wait(deadline - time.time())
                else:
                    # Python condition variables cannot be interrupted unless
                    # they have a timeout. This can make lit unresponsive to
                    # KeyboardInterrupt, so do a busy wait with a timeout.
                    while not a.ready():
                        a.wait(1)
                if not a.successful():
                    a.get() # Exceptions raised here come from the worker.
                if self.hit_max_failures:
                    break
        except:
            # Stop the workers and wait for any straggling results to come in
            # if we exited without waiting on every async result.
            pool.terminate()
            raise
        finally:
            pool.join()

    def execute_tests(self, display, jobs, max_time=None):
        """
        execute_tests(display, jobs, [max_time])

        Execute each of the tests in the run, using up to jobs number of
        parallel tasks, and inform the display of each individual result. The
        provided tests should be a subset of the tests available in this run
        object.

        If max_time is non-None, it should be a time in seconds after which to
        stop executing tests.

        The display object will have its update method called with each test as
        it is completed. The calls are guaranteed to be locked with respect to
        one another, but are *not* guaranteed to be called on the same thread as
        this method was invoked on.

        Upon completion, each test in the run will have its result
        computed. Tests which were not actually executed (for any reason) will
        be given an UNRESOLVED result.
        """
        # Don't do anything if we aren't going to run any tests.
        if not self.tests:
            return

        # Save the display object on the runner so that we can update it from
        # our task completion callback.
        self.display = display

        self.failure_count = 0
        self.hit_max_failures = False
        if jobs == 1:
            for test_index, test in enumerate(self.tests):
                lit.worker._execute_test(test, self.lit_config)
                self.consume_test_result((test_index, test))
                if self.hit_max_failures:
                    break
        else:
            self.execute_tests_in_pool(jobs, max_time)

        # Mark any tests that weren't run as UNRESOLVED.
        for test in self.tests:
            if test.result is None:
                test.setResult(lit.Test.Result(lit.Test.UNRESOLVED, '', 0.0))

    def consume_test_result(self, pool_result):
        """Test completion callback for lit.worker.run_one_test

        Updates the test result status in the parent process. Each task in the
        pool returns the test index and the result, and we use the index to look
        up the original test object. Also updates the progress bar as tasks
        complete.
        """
        # Don't add any more test results after we've hit the maximum failure
        # count.  Otherwise we're racing with the main thread, which is going
        # to terminate the process pool soon.
        if self.hit_max_failures:
            return

        (test_index, test_with_result) = pool_result
        # Update the parent process copy of the test. This includes the result,
        # XFAILS, REQUIRES, and UNSUPPORTED statuses.
        assert self.tests[test_index].file_path == test_with_result.file_path, \
                "parent and child disagree on test path"
        self.tests[test_index] = test_with_result
        self.display.update(test_with_result)

        # If we've finished all the tests or too many tests have failed, notify
        # the main thread that we've stopped testing.
        self.failure_count += (test_with_result.result.code == lit.Test.FAIL)
        if self.lit_config.maxFailures and \
                self.failure_count == self.lit_config.maxFailures:
            self.hit_max_failures = True
