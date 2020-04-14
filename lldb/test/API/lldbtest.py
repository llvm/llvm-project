from __future__ import absolute_import
import os
import tempfile
import subprocess
import sys

import lit.Test
import lit.TestRunner
import lit.util
from lit.formats.base import TestFormat

def getBuildDir(cmd):
    found = False
    for arg in cmd:
        if found:
            return arg
        if arg == '--build-dir':
            found = True
    return None

def mkdir_p(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if not os.path.isdir(path):
        raise OSError(errno.ENOTDIR, "%s is not a directory"%path)

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

        if not test.config.lldb_enable_python:
            return (lit.Test.UNSUPPORTED, 'Python module disabled')

        if test.config.unsupported:
            return (lit.Test.UNSUPPORTED, 'Test is unsupported')

        testPath, testFile = os.path.split(test.getSourcePath())
        # On Windows, the system does not always correctly interpret
        # shebang lines.  To make sure we can execute the tests, add
        # python exe as the first parameter of the command.
        cmd = [sys.executable] + self.dotest_cmd + [testPath, '-p', testFile]

        builddir = getBuildDir(cmd)
        mkdir_p(builddir)

        # The macOS system integrity protection (SIP) doesn't allow injecting
        # libraries into system binaries, but this can be worked around by
        # copying the binary into a different location.
        if 'DYLD_INSERT_LIBRARIES' in test.config.environment and \
                (sys.executable.startswith('/System/') or \
                sys.executable.startswith('/usr/bin/')):
            copied_python = os.path.join(builddir, 'copied-system-python')
            if not os.path.isfile(copied_python):
                import shutil, subprocess
                python = subprocess.check_output([
                    sys.executable,
                    '-c',
                    'import sys; print(sys.executable)'
                ]).decode('utf-8').strip()
                shutil.copy(python, copied_python)
            cmd[0] = copied_python

        if 'lldb-repro-capture' in test.config.available_features or \
           'lldb-repro-replay' in test.config.available_features:
            reproducer_root = os.path.join(builddir, 'reproducers')
            mkdir_p(reproducer_root)
            reproducer_path = os.path.join(reproducer_root, testFile)
            if 'lldb-repro-capture' in test.config.available_features:
                cmd.extend(['--capture-path', reproducer_path])
            else:
                cmd.extend(['--replay-path', reproducer_path])

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

        if exitCode:
            if 'XPASS:' in out or 'XPASS:' in err:
                return lit.Test.XPASS, output

            # Otherwise this is just a failure.
            return lit.Test.FAIL, output

        has_unsupported_tests = 'UNSUPPORTED:' in out or 'UNSUPPORTED:' in err
        has_passing_tests = 'PASS:' in out or 'PASS:' in err
        if has_unsupported_tests and not has_passing_tests:
            return lit.Test.UNSUPPORTED, output

        passing_test_line = 'RESULT: PASSED'
        if passing_test_line not in out and passing_test_line not in err:
            return lit.Test.UNRESOLVED, output

        return lit.Test.PASS, output
