import collections
import json
import os
import re
import operator

import lit.Test
import lit.TestRunner
import lit.util
from lit.formats.base import TestFormat

import lldbflakes

# Tests discovered by this format in the main lit process. lit copies each
# worker's result back onto these objects, so they can be summarized at exit.
_discovered_tests = []

# Result codes dotest can report for an individual test method.
DOTEST_CODES = [
    lit.Test.SKIPPED,
    lit.Test.UNSUPPORTED,
    lit.Test.PASS,
    lit.Test.XFAIL,
    lit.Test.FAIL,
    lit.Test.XPASS,
]
# Result codes for which the test body did not execute.
NOT_RAN_CODES = {lit.Test.SKIPPED.name, lit.Test.UNSUPPORTED.name}


def _parse_debug_info_breakdown(output):
    match = re.search(r"^Debug info breakdown: (.*)$", output or "", re.MULTILINE)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except ValueError:
        return {}


def print_debug_info_report():
    """Print a table of API test results per debug info variant."""
    totals = collections.defaultdict(collections.Counter)
    for test in _discovered_tests:
        if test.result is None:
            continue
        for variant, counts in _parse_debug_info_breakdown(test.result.output).items():
            totals[variant].update(counts)
    if not totals:
        return

    def pct(num, den):
        return "%.1f%%" % (100.0 * num / den) if den else "-"

    # dotest reports outcomes using lit's result code names. Always show every
    # code so the table layout is the same from run to run.
    header = ["Variant", "Total"] + [code.label for code in DOTEST_CODES]
    header += ["Ran", "Passed (of ran)"]
    rows = []
    for variant in sorted(totals):
        counts = totals[variant]
        total = sum(counts.values())
        ran = total - sum(counts[name] for name in NOT_RAN_CODES)
        rows.append(
            [variant, str(total)]
            + [str(counts[code.name]) for code in DOTEST_CODES]
            + [pct(ran, total), pct(counts[lit.Test.PASS.name], ran)]
        )

    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]

    def fmt(row):
        return "  ".join(
            cell.ljust(w) if i == 0 else cell.rjust(w)
            for i, (cell, w) in enumerate(zip(row, widths))
        )

    print("\nLLDB API test results per debug info variant:")
    print(fmt(header))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


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
                    test = lit.Test.Test(
                        testSuite, path_in_suite + (filename,), localConfig
                    )
                    _discovered_tests.append(test)
                    yield test

    def getTestsForPath(self, testSuite, path_in_suite, litConfig, localConfig):
        # Used when a test file is named directly on the lit command line.
        for test in super().getTestsForPath(
            testSuite, path_in_suite, litConfig, localConfig
        ):
            _discovered_tests.append(test)
            yield test

    def execute(self, test, litConfig):
        return lldbflakes.execute_with_reruns(
            test, lambda: lit.Test.Result(*self._execute_once(test, litConfig))
        )

    def _execute_once(self, test, litConfig):
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

        if test.config.maxIndividualTestTime > 0:
            cmd += ["--timeout", str(test.config.maxIndividualTestTime)]

        launcher = getattr(test.config, "lldb_launcher", None)
        if launcher:
            cmd = [launcher] + cmd

        if isLuaTest:
            cmd.extend(["--env", "LUA_EXECUTABLE=%s" % test.config.lua_executable])
            cmd.extend(["--env", "LLDB_LUA_CPATH=%s" % test.config.lldb_lua_cpath])

        timeoutInfo = None
        try:
            out, err, exitCode = lit.util.executeCommand(
                cmd,
                env=test.config.environment,
                timeout=test.config.maxIndividualTestTime,
            )
        except lit.util.ExecuteCommandTimeoutException as e:
            out = e.out
            err = e.err
            exitCode = e.exitCode
            timeoutInfo = "Reached timeout of {} seconds".format(
                test.config.maxIndividualTestTime
            )

        # On Windows, dotest writes CRLF line endings. Normalize them here, as
        # lit's own text-mode output would otherwise turn them into CRCRLF.
        out = out.replace("\r\n", "\n") if out else out
        err = err.replace("\r\n", "\n") if err else err

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
        result_regex = r"^(?:OK|FAILED)(?: \((.*)\))?\r?$"
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

        only_skipped = 0
        breakdown = re.search(
            r"^Skip breakdown \(unsupported=(\d+), skipped=(\d+)\)\r?$",
            err,
            re.MULTILINE,
        )
        if breakdown:
            only_skipped = int(breakdown.group(2))

        non_pass = (
            failures + errors + skipped + expected_failures + unexpected_successes
        )
        passes = num_ran - non_pass

        if exitCode:
            # Aggregate the tests results with the following precedence:
            # UNRESOLVED > FAIL > XPASS
            if errors > 0:
                return lit.Test.UNRESOLVED, output
            if failures > 0:
                return lit.Test.FAIL, output
            return lit.Test.XPASS, output
        else:
            # Aggregate the tests results with the following precedence:
            # PASS > XFAIL > SKIPPED > UNSUPPORTED
            if passes > 0:
                return lit.Test.PASS, output
            if expected_failures > 0:
                return lit.Test.XFAIL, output
            if only_skipped > 0:
                return lit.Test.SKIPPED, output
            return lit.Test.UNSUPPORTED, output
