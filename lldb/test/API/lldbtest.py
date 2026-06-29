import collections
import os
import platform
import re
import operator
import shutil
import signal
import subprocess
import threading

import lit.Test
import lit.TestRunner
import lit.util
from lit.formats.base import TestFormat


def _sample_hung_process_tree(pid: int) -> str:
    """Capture of all-thread backtraces for the process ``pid`` and
    all of its descendants, returned as a single string. Return an empty string
    if we cannot capture a backtrace.
    """
    # The `sample` tool is macOS-only and is the only sampling approach we
    # current support.
    if platform.system() != "Darwin":
        return ""

    sampler = shutil.which("sample")
    if not sampler:
        return ""

    try:
        import psutil
    except ImportError:
        return ""

    try:
        root = psutil.Process(pid)
        procs = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return ""

    chunks = []
    for proc in procs:
        try:
            proc_pid, proc_name = proc.pid, proc.name()
        except psutil.NoSuchProcess:
            continue
        try:
            # "sample <pid> <duration_secs>" takes a quick snapshot of every
            # thread. -mayDie in the unlikely case the process actually finished
            # while sampling.
            result = subprocess.run(
                [sampler, str(proc_pid), "1", "-mayDie"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            text = result.stdout.decode("utf-8", errors="replace")
        except Exception as e:
            text = "<failed to sample: {}>".format(e)
        chunks.append(
            "===== Backtrace of hung process {} ({}) =====\n{}".format(
                proc_pid, proc_name, text
            )
        )
    return "\n".join(chunks)


class LLDBTest(TestFormat):
    def __init__(self, dotest_cmd):
        self.dotest_cmd = dotest_cmd

    def executeCommand(self, command, env, timeout):
        """Like lit.util.executeCommand, but when ``timeout`` is hit it captures
        backtraces of the (hung) process tree before killing it.

        Returns (out, err, exitCode, timed_out, sample_output).
        """
        p = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            close_fds=lit.util.kUseCloseFDs,
        )

        # Arrays so we can modify them by reference from on_timeout.
        timed_out = [False]
        sample_output = [""]
        timer = None

        if timeout and timeout > 0:

            def on_timeout():
                timed_out[0] = True
                # Snapshot the stacks of the hung process tree, then kill it
                # (process and all children, matching lit's behavior).
                sample_output[0] = _sample_hung_process_tree(p.pid)
                lit.util.killProcessAndChildren(p.pid)

            timer = threading.Timer(timeout, on_timeout)
            timer.start()

        try:
            out, err = p.communicate()
            exitCode = p.wait()
        finally:
            if timer is not None:
                timer.cancel()

        out = out.decode("utf-8", errors="replace")
        err = err.decode("utf-8", errors="replace")

        # Detect Ctrl-C in the subprocess.
        if not timed_out[0] and exitCode == -signal.SIGINT:
            raise KeyboardInterrupt

        return out, err, exitCode, timed_out[0], sample_output[0]

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

        launcher = getattr(test.config, "lldb_launcher", None)
        if launcher:
            cmd = [launcher] + cmd

        if isLuaTest:
            cmd.extend(["--env", "LUA_EXECUTABLE=%s" % test.config.lua_executable])
            cmd.extend(["--env", "LLDB_LUA_CPATH=%s" % test.config.lldb_lua_cpath])

        timeoutInfo = None
        out, err, exitCode, timedOut, sampleOutput = self.executeCommand(
            cmd,
            env=test.config.environment,
            timeout=test.config.maxIndividualTestTime,
        )
        if timedOut:
            timeoutInfo = "Reached timeout of {} seconds".format(
                test.config.maxIndividualTestTime
            )

        output = """Script:\n--\n%s\n--\nExit Code: %d\n""" % (" ".join(cmd), exitCode)
        if timeoutInfo is not None:
            output += """Timeout: %s\n""" % (timeoutInfo,)
        output += "\n"

        if out:
            output += """Command Output (stdout):\n--\n%s\n--\n""" % (out,)
        if err:
            output += """Command Output (stderr):\n--\n%s\n--\n""" % (err,)
        if sampleOutput:
            output += (
                """Backtraces of hung process tree (captured on timeout):\n"""
                """--\n%s\n--\n""" % (sampleOutput,)
            )

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
            # PASS > XFAIL > UNSUPPORTED
            if passes > 0:
                return lit.Test.PASS, output
            if expected_failures > 0:
                return lit.Test.XFAIL, output
            return lit.Test.UNSUPPORTED, output
