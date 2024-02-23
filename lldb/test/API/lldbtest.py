import collections
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

    def getTestsInDirectory(self, testSuite, path_in_suite, litConfig, localConfig):
        source_path = testSuite.getSourcePath(path_in_suite)
        for filename in os.listdir(source_path):
            # Ignore dot files and excluded tests.
            if filename.startswith(".") or filename in localConfig.excludes:
                continue

            # Ignore files that don't start with 'Test'.
            if not filename.startswith("Test"):
                continue

            filepath = os.path.join(source_path, filename)
            if not os.path.isdir(filepath):
                base, ext = os.path.splitext(filename)
                if ext in localConfig.suffixes:
                    yield lit.Test.Test(
                        testSuite, path_in_suite + (filename,), localConfig
                    )

    def execute(self, test, litConfig):
        if litConfig.noExecute:
            return lit.Test.PASS, ""

        if not getattr(test.config, "lldb_enable_python", False):
            return (lit.Test.UNSUPPORTED, "Python module disabled")

        if test.config.unsupported:
            return (lit.Test.UNSUPPORTED, "Test is unsupported")

        testPath, testFile = os.path.split(test.getSourcePath())

        # The Python used to run lit can be different from the Python LLDB was
        # build with.
        executable = test.config.python_executable

        isLuaTest = testFile == test.config.lua_test_entry

        # On Windows, the system does not always correctly interpret
        # shebang lines.  To make sure we can execute the tests, add
        # python exe as the first parameter of the command.
        cmd = [executable] + self.dotest_cmd + [testPath, "-p", testFile]

        if isLuaTest:
            luaExecutable = test.config.lua_executable
            cmd.extend(["--env", "LUA_EXECUTABLE=%s" % luaExecutable])

        timeoutInfo = None
        try:
            out, err, exitCode = lit.util.executeCommand(
                cmd,
                env=test.config.environment,
                timeout=litConfig.maxIndividualTestTime,
            )
        except lit.util.ExecuteCommandTimeoutException as e:
            out = e.out
            err = e.err
            exitCode = e.exitCode
            timeoutInfo = "Reached timeout of {} seconds".format(
                litConfig.maxIndividualTestTime
            )

        output = """Script:\n--\n%s\n--\nExit Code: %d\n""" % (" ".join(cmd), exitCode)
        if timeoutInfo is not None:
            output += """Timeout: %s\n""" % (timeoutInfo,)
        output += "\n"

        if out:
            output += """Command Output (stdout):\n--\n%s\n--\n""" % (out,)
        if err:
            output += """Command Output (stderr):\n--\n%s\n--\n""" % (err,)

        if timeoutInfo:
            return lit.Test.TIMEOUT, output

        # Parse the dotest output from stderr. First get the # of total tests, in order to infer the # of passes.
        # Example: "Ran 5 tests in 0.042s"
        num_ran_regex = r"^Ran (\d+) tests? in "
        num_ran_results = re.search(num_ran_regex, err, re.MULTILINE)

        # If parsing fails mark this test as unresolved.
        if not num_ran_results:
            return lit.Test.UNRESOLVED, output
        num_ran = int(num_ran_results.group(1))

        # Then look for a detailed summary, which is OK or FAILED followed by optional details.
        # Example: "OK (skipped=1, expected failures=1)"
        # Example: "FAILED (failures=3)"
        # Example: "OK"
        result_regex = r"^(?:OK|FAILED)(?: \((.*)\))?$"
        results = re.search(result_regex, err, re.MULTILINE)

        # If parsing fails mark this test as unresolved.
        if not results:
            return lit.Test.UNRESOLVED, output

        details = results.group(1)
        parsed_details = collections.defaultdict(int)
        if details:
            for detail in details.split(", "):
                detail_parts = detail.split("=")
                if len(detail_parts) != 2:
                    return lit.Test.UNRESOLVED, output
                parsed_details[detail_parts[0]] = int(detail_parts[1])

        failures = parsed_details["failures"]
        errors = parsed_details["errors"]
        skipped = parsed_details["skipped"]
        expected_failures = parsed_details["expected failures"]
        unexpected_successes = parsed_details["unexpected successes"]

        non_pass = (
            failures + errors + skipped + expected_failures + unexpected_successes
        )
        passes = num_ran - non_pass

        if exitCode:
            # Mark this test as FAIL if at least one test failed.
            if failures > 0:
                return lit.Test.FAIL, output
            lit_results = [
                (failures, lit.Test.FAIL),
                (errors, lit.Test.UNRESOLVED),
                (unexpected_successes, lit.Test.XPASS),
            ]
        else:
            # Mark this test as PASS if at least one test passed.
            if passes > 0:
                return lit.Test.PASS, output
            lit_results = [
                (passes, lit.Test.PASS),
                (skipped, lit.Test.UNSUPPORTED),
                (expected_failures, lit.Test.XFAIL),
            ]

        # Return the lit result code with the maximum occurrence. Only look at
        # the first element and rely on the original order to break ties.
        return max(lit_results, key=operator.itemgetter(0))[1], output
