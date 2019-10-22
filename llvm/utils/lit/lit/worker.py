"""
The functions in this module are meant to run on a separate worker process.
Exception: in single process mode _execute is called directly.

For efficiency, we copy all data needed to execute all tests into each worker
and store it in global variables. This reduces the cost of each task.
"""
import time
import traceback

import lit.Test
import lit.util

_lit_config = None
_parallelism_semaphores = None

def initialize(lit_config, parallelism_semaphores):
    """Copy data shared by all test executions into worker processes"""
    global _lit_config
    global _parallelism_semaphores
    _lit_config = lit_config
    _parallelism_semaphores = parallelism_semaphores

def execute(test):
    """Run one test in a multiprocessing.Pool

    Side effects in this function and functions it calls are not visible in the
    main lit process.

    Arguments and results of this function are pickled, so they should be cheap
    to copy.
    """
    try:
        return _execute_in_parallelism_group(test, _lit_config,
                                             _parallelism_semaphores)
    except KeyboardInterrupt:
        # If a worker process gets an interrupt, abort it immediately.
        lit.util.abort_now()
    except:
        traceback.print_exc()

def _execute_in_parallelism_group(test, lit_config, parallelism_semaphores):
    pg = test.config.parallelism_group
    if callable(pg):
        pg = pg(test)

    if pg:
        semaphore = parallelism_semaphores[pg]
        try:
            semaphore.acquire()
            return _execute(test, lit_config)
        finally:
            semaphore.release()
    else:
        return _execute(test, lit_config)


def _execute(test, lit_config):
    """Execute one test"""
    start = time.time()
    result = _execute_test_handle_errors(test, lit_config)
    end = time.time()

    result.elapsed = end - start
    resolve_result_code(result, test)

    return result


# TODO(yln): is this the right place to deal with this?
# isExpectedToFail() only works after the test has been executed.
def resolve_result_code(result, test):
    try:
        expected_to_fail = test.isExpectedToFail()
    except ValueError as e:
        # Syntax error in an XFAIL line.
        result.code = lit.Test.UNRESOLVED
        result.output = str(e)
    else:
        if expected_to_fail:
            # pass -> unexpected pass
            if result.code is lit.Test.PASS:
                result.code = lit.Test.XPASS
            # fail -> expected fail
            if result.code is lit.Test.FAIL:
                result.code = lit.Test.XFAIL


def _execute_test_handle_errors(test, lit_config):
    try:
        return _adapt_result(test.config.test_format.execute(test, lit_config))
    except KeyboardInterrupt:
        raise
    except:
        if lit_config.debug:
            raise
        output = 'Exception during script execution:\n'
        output += traceback.format_exc()
        output += '\n'
        return lit.Test.Result(lit.Test.UNRESOLVED, output)


# Support deprecated result from execute() which returned the result
# code and additional output as a tuple.
def _adapt_result(result):
    if isinstance(result, lit.Test.Result):
        return result
    assert isinstance(result, tuple)
    code, output = result
    return lit.Test.Result(code, output)
