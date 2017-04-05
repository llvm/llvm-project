import os
import sys
import threading
import time
import traceback
try:
    import Queue as queue
except ImportError:
    import queue

try:
    import win32api
except ImportError:
    win32api = None

try:
    import multiprocessing
except ImportError:
    multiprocessing = None

import lit.Test

###
# Test Execution Implementation

class LockedValue(object):
    def __init__(self, value):
        self.lock = threading.Lock()
        self._value = value

    def _get_value(self):
        self.lock.acquire()
        try:
            return self._value
        finally:
            self.lock.release()

    def _set_value(self, value):
        self.lock.acquire()
        try:
            self._value = value
        finally:
            self.lock.release()

    value = property(_get_value, _set_value)

class TestProvider(object):
    def __init__(self, queue_impl, canceled_flag):
        self.canceled_flag = canceled_flag

        # Create a shared queue to provide the test indices.
        self.queue = queue_impl()

    def queue_tests(self, tests, num_jobs):
        for i in range(len(tests)):
            self.queue.put(i)
        for i in range(num_jobs):
            self.queue.put(None)

    def cancel(self):
        self.canceled_flag.value = 1

    def get(self):
        # Check if we are canceled.
        if self.canceled_flag.value:
          return None

        # Otherwise take the next test.
        return self.queue.get()

class Tester(object):
    def __init__(self, run_instance, provider, consumer):
        self.run_instance = run_instance
        self.provider = provider
        self.consumer = consumer

    def run(self):
        while True:
            item = self.provider.get()
            if item is None:
                break
            self.run_test(item)
        self.consumer.task_finished()

    def run_test(self, test_index):
        test = self.run_instance.tests[test_index]
        try:
            execute_test(test, self.run_instance.lit_config,
                         self.run_instance.parallelism_semaphores)
        except KeyboardInterrupt:
            # This is a sad hack. Unfortunately subprocess goes
            # bonkers with ctrl-c and we start forking merrily.
            print('\nCtrl-C detected, goodbye.')
            sys.stdout.flush()
            os.kill(0,9)
        self.consumer.update(test_index, test)

class ThreadResultsConsumer(object):
    def __init__(self, display):
        self.display = display
        self.lock = threading.Lock()

    def update(self, test_index, test):
        self.lock.acquire()
        try:
            self.display.update(test)
        finally:
            self.lock.release()

    def task_finished(self):
        pass

    def handle_results(self):
        pass

class MultiprocessResultsConsumer(object):
    def __init__(self, run, display, num_jobs):
        self.run = run
        self.display = display
        self.num_jobs = num_jobs
        self.queue = multiprocessing.Queue()

    def update(self, test_index, test):
        # This method is called in the child processes, and communicates the
        # results to the actual display implementation via an output queue.
        self.queue.put((test_index, test.result))

    def task_finished(self):
        # This method is called in the child processes, and communicates that
        # individual tasks are complete.
        self.queue.put(None)

    def handle_results(self):
        # This method is called in the parent, and consumes the results from the
        # output queue and dispatches to the actual display. The method will
        # complete after each of num_jobs tasks has signalled completion.
        completed = 0
        while completed != self.num_jobs:
            # Wait for a result item.
            item = self.queue.get()
            if item is None:
                completed += 1
                continue

            # Update the test result in the parent process.
            index,result = item
            test = self.run.tests[index]
            test.result = result

            self.display.update(test)

def run_one_tester(run, provider, display):
    tester = Tester(run, provider, display)
    tester.run()

###

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

def handleFailures(provider, consumer, maxFailures):
    consumer.display = _Display(consumer.display, provider, maxFailures)

def execute_test(test, lit_config, parallelism_semaphores):
    """Execute one test"""
    pg = test.config.parallelism_group
    if callable(pg):
        pg = pg(test)

    result = None
    semaphore = None
    try:
        if pg:
            semaphore = parallelism_semaphores[pg]
        if semaphore:
            semaphore.acquire()
        start_time = time.time()
        result = test.config.test_format.execute(test, lit_config)
        # Support deprecated result from execute() which returned the result
        # code and additional output as a tuple.
        if isinstance(result, tuple):
            code, output = result
            result = lit.Test.Result(code, output)
        elif not isinstance(result, lit.Test.Result):
            raise ValueError("unexpected result from test execution")
        result.elapsed = time.time() - start_time
    except KeyboardInterrupt:
        raise
    except:
        if lit_config.debug:
            raise
        output = 'Exception during script execution:\n'
        output += traceback.format_exc()
        output += '\n'
        result = lit.Test.Result(lit.Test.UNRESOLVED, output)
    finally:
        if semaphore:
            semaphore.release()

    test.setResult(result)

class Run(object):
    """
    This class represents a concrete, configured testing run.
    """

    def __init__(self, lit_config, tests):
        self.lit_config = lit_config
        self.tests = tests

    def execute_test(self, test):
        return execute_test(test, self.lit_config, self.parallelism_semaphores)

    def execute_tests(self, display, jobs, max_time=None,
                      execution_strategy=None):
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

        if execution_strategy == 'PROCESS_POOL':
            self.execute_tests_with_mp_pool(display, jobs, max_time)
            return
        # FIXME: Standardize on the PROCESS_POOL execution strategy and remove
        # the other two strategies.

        use_processes = execution_strategy == 'PROCESSES'

        # Choose the appropriate parallel execution implementation.
        consumer = None
        if jobs != 1 and use_processes and multiprocessing:
            try:
                task_impl = multiprocessing.Process
                queue_impl = multiprocessing.Queue
                sem_impl = multiprocessing.Semaphore
                canceled_flag =  multiprocessing.Value('i', 0)
                consumer = MultiprocessResultsConsumer(self, display, jobs)
            except:
                # multiprocessing fails to initialize with certain OpenBSD and
                # FreeBSD Python versions: http://bugs.python.org/issue3770
                # Unfortunately the error raised also varies by platform.
                self.lit_config.note('failed to initialize multiprocessing')
                consumer = None
        if not consumer:
            task_impl = threading.Thread
            queue_impl = queue.Queue
            sem_impl = threading.Semaphore
            canceled_flag = LockedValue(0)
            consumer = ThreadResultsConsumer(display)

        self.parallelism_semaphores = {k: sem_impl(v)
            for k, v in self.lit_config.parallelism_groups.items()}

        # Create the test provider.
        provider = TestProvider(queue_impl, canceled_flag)
        handleFailures(provider, consumer, self.lit_config.maxFailures)

        # Putting tasks into the threading or multiprocessing Queue may block,
        # so do it in a separate thread.
        # https://docs.python.org/2/library/multiprocessing.html
        # e.g: On Mac OS X, we will hang if we put 2^15 elements in the queue
        # without taking any out.
        queuer = task_impl(target=provider.queue_tests, args=(self.tests, jobs))
        queuer.start()

        # Install a console-control signal handler on Windows.
        if win32api is not None:
            def console_ctrl_handler(type):
                provider.cancel()
                return True
            win32api.SetConsoleCtrlHandler(console_ctrl_handler, True)

        # Install a timeout handler, if requested.
        if max_time is not None:
            def timeout_handler():
                provider.cancel()
            timeout_timer = threading.Timer(max_time, timeout_handler)
            timeout_timer.start()

        # If not using multiple tasks, just run the tests directly.
        if jobs == 1:
            run_one_tester(self, provider, consumer)
        else:
            # Otherwise, execute the tests in parallel
            self._execute_tests_in_parallel(task_impl, provider, consumer, jobs)

        queuer.join()

        # Cancel the timeout handler.
        if max_time is not None:
            timeout_timer.cancel()

        # Update results for any tests which weren't run.
        for test in self.tests:
            if test.result is None:
                test.setResult(lit.Test.Result(lit.Test.UNRESOLVED, '', 0.0))

    def _execute_tests_in_parallel(self, task_impl, provider, consumer, jobs):
        # Start all of the tasks.
        tasks = [task_impl(target=run_one_tester,
                           args=(self, provider, consumer))
                 for i in range(jobs)]
        for t in tasks:
            t.start()

        # Allow the consumer to handle results, if necessary.
        consumer.handle_results()

        # Wait for all the tasks to complete.
        for t in tasks:
            t.join()

    def execute_tests_with_mp_pool(self, display, jobs, max_time=None):
        # Don't do anything if we aren't going to run any tests.
        if not self.tests or jobs == 0:
            return

        # Set up semaphores to limit parallelism of certain classes of tests.
        # For example, some ASan tests require lots of virtual memory and run
        # faster with less parallelism on OS X.
        self.parallelism_semaphores = \
                {k: multiprocessing.Semaphore(v) for k, v in
                 self.lit_config.parallelism_groups.items()}

        # Save the display object on the runner so that we can update it from
        # our task completion callback.
        self.display = display

        # Start a process pool. Copy over the data shared between all test runs.
        pool = multiprocessing.Pool(jobs, worker_initializer,
                                    (self.lit_config,
                                     self.parallelism_semaphores))

        # Install a console-control signal handler on Windows.
        if win32api is not None:
            def console_ctrl_handler(type):
                print('\nCtrl-C detected, terminating.')
                pool.terminate()
                pool.join()
                os.kill(0,9)
                return True
            win32api.SetConsoleCtrlHandler(console_ctrl_handler, True)

        # FIXME: Implement max_time using .wait() timeout argument and a
        # deadline.

        try:
            async_results = [pool.apply_async(worker_run_one_test,
                                              args=(test_index, test),
                                              callback=self.consume_test_result)
                             for test_index, test in enumerate(self.tests)]

            # Wait for all results to come in. The callback that runs in the
            # parent process will update the display.
            for a in async_results:
                a.wait()
                if not a.successful():
                    a.get() # Exceptions raised here come from the worker.
        finally:
            pool.terminate()
            pool.join()

        # Mark any tests that weren't run as UNRESOLVED.
        for test in self.tests:
            if test.result is None:
                test.setResult(lit.Test.Result(lit.Test.UNRESOLVED, '', 0.0))

    def consume_test_result(self, pool_result):
        """Test completion callback for worker_run_one_test

        Updates the test result status in the parent process. Each task in the
        pool returns the test index and the result, and we use the index to look
        up the original test object. Also updates the progress bar as tasks
        complete.
        """
        (test_index, test_with_result) = pool_result
        # Update the parent process copy of the test. This includes the result,
        # XFAILS, REQUIRES, and UNSUPPORTED statuses.
        assert self.tests[test_index].file_path == test_with_result.file_path, \
                "parent and child disagree on test path"
        self.tests[test_index] = test_with_result
        self.display.update(test_with_result)

child_lit_config = None
child_parallelism_semaphores = None

def worker_initializer(lit_config, parallelism_semaphores):
    """Copy expensive repeated data into worker processes"""
    global child_lit_config
    child_lit_config = lit_config
    global child_parallelism_semaphores
    child_parallelism_semaphores = parallelism_semaphores

def worker_run_one_test(test_index, test):
    """Run one test in a multiprocessing.Pool

    Side effects in this function and functions it calls are not visible in the
    main lit process.

    Arguments and results of this function are pickled, so they should be cheap
    to copy. For efficiency, we copy all data needed to execute all tests into
    each worker and store it in the child_* global variables. This reduces the
    cost of each task.

    Returns an index and a Result, which the parent process uses to update
    the display.
    """
    try:
        execute_test(test, child_lit_config, child_parallelism_semaphores)
        return (test_index, test)
    except KeyboardInterrupt as e:
        # This is a sad hack. Unfortunately subprocess goes
        # bonkers with ctrl-c and we start forking merrily.
        print('\nCtrl-C detected, goodbye.')
        traceback.print_exc()
        sys.stdout.flush()
        os.kill(0,9)
    except:
        traceback.print_exc()
