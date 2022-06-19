from __future__ import absolute_import
import os
import re
import operator

import lit.Test
import lit.TestRunner
import lit.util
from lit.formats.base import TestFormat


class LLDBTest(TestFormat):
    def __init__(self, dotest_cmd):
        self.dotest_cmd = dotest_cmd

    def getTestsInDirectory(self, testSuite, path_in_suite, litConfig,
                            localConfig):
        source_path = testSuite.getSourcePath(path_in_suite)
        for filename in os.listdir(source_path):
            # Ignore dot files and excluded tests.
            if (filename.startswith('.') or filename in localConfig.excludes):
                continue

            # Ignore files that don't start with 'Test'.
            if not filename.startswith('Test'):
                continue

            filepath = os.path.join(source_path, filename)
            if not os.path.isdir(filepath):
                base, ext = os.path.splitext(filename)
                if ext in localConfig.suffixes:
                    yield lit.Test.Test(testSuite, path_in_suite +
                                        (filename, ), localConfig)

    def execute(self, test, litConfig):
        if litConfig.noExecute:
            return lit.Test.PASS, ''

        if not getattr(test.config, 'lldb_enable_python', False):
            return (lit.Test.UNSUPPORTED, 'Python module disabled')

        if test.config.unsupported:
            return (lit.Test.UNSUPPORTED, 'Test is unsupported')

        testPath, testFile = os.path.split(test.getSourcePath())

        # The Python used to run lit can be different from the Python LLDB was
        # build with.
        executable = test.config.python_executable

        isLuaTest = testFile == test.config.lua_test_entry

        # On Windows, the system does not always correctly interpret
        # shebang lines.  To make sure we can execute the tests, add
        # python exe as the first parameter of the command.
        cmd = [executable] + self.dotest_cmd + [testPath, '-p', testFile]

        if isLuaTest:
            luaExecutable = test.config.lua_executable
            cmd.extend(['--env', 'LUA_EXECUTABLE=%s' % luaExecutable])

        timeoutInfo = None
        try:
            out, err, exitCode = lit.util.executeCommand(
                cmd,
                env=test.config.environment,
                timeout=litConfig.maxIndividualTestTime)
        except lit.util.ExecuteCommandTimeoutException as e:
            out = e.out
            err = e.err
            exitCode = e.exitCode
            timeoutInfo = 'Reached timeout of {} seconds'.format(
                litConfig.maxIndividualTestTime)

        output = """Script:\n--\n%s\n--\nExit Code: %d\n""" % (
            ' '.join(cmd), exitCode)
        if timeoutInfo is not None:
            output += """Timeout: %s\n""" % (timeoutInfo,)
        output += "\n"

        if out:
            output += """Command Output (stdout):\n--\n%s\n--\n""" % (out,)
        if err:
            output += """Command Output (stderr):\n--\n%s\n--\n""" % (err,)

        if timeoutInfo:
            return lit.Test.TIMEOUT, output

        # Parse the dotest output from stderr.
        result_regex = r"\((\d+) passes, (\d+) failures, (\d+) errors, (\d+) skipped, (\d+) expected failures, (\d+) unexpected successes\)"
        results = re.search(result_regex, err)

        # If parsing fails mark this test as unresolved.
        if not results:
            return lit.Test.UNRESOLVED, output

        passes = int(results.group(1))
        failures = int(results.group(2))
        errors = int(results.group(3))
        skipped = int(results.group(4))
        expected_failures = int(results.group(5))
        unexpected_successes = int(results.group(6))

        if exitCode:
            # Mark this test as FAIL if at least one test failed.
            if failures > 0:
                return lit.Test.FAIL, output
            lit_results = [(failures, lit.Test.FAIL),
                           (errors, lit.Test.UNRESOLVED),
                           (unexpected_successes, lit.Test.XPASS)]
        else:
            # Mark this test as PASS if at least one test passed.
            if passes > 0:
                return lit.Test.PASS, output
            lit_results = [(passes, lit.Test.PASS),
                           (skipped, lit.Test.UNSUPPORTED),
                           (expected_failures, lit.Test.XFAIL)]

        # Return the lit result code with the maximum occurrence. Only look at
        # the first element and rely on the original order to break ties.
        return max(lit_results, key=operator.itemgetter(0))[1], output
