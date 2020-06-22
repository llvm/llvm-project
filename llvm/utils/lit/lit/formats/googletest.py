from __future__ import absolute_import
import os
import subprocess
import sys

import lit.Test
import lit.TestRunner
import lit.util
from .base import TestFormat

kIsWindows = sys.platform in ['win32', 'cygwin']

class GoogleTest(TestFormat):
    def __init__(self, test_sub_dirs, test_suffix):
        self.test_sub_dirs = str(test_sub_dirs).split(';')

        # On Windows, assume tests will also end in '.exe'.
        exe_suffix = str(test_suffix)
        if kIsWindows:
            exe_suffix += '.exe'

        # Also check for .py files for testing purposes.
        self.test_suffixes = {exe_suffix, test_suffix + '.py'}

    def getGTestTests(self, path, litConfig, localConfig):
        """getGTestTests(path) - [name]

        Return the tests available in gtest executable.

        Args:
          path: String path to a gtest executable
          litConfig: LitConfig instance
          localConfig: TestingConfig instance"""

        list_test_cmd = self.maybeAddPythonToCmd([path, '--gtest_list_tests'])

        try:
            output = subprocess.check_output(list_test_cmd,
                                             env=localConfig.environment)
        except subprocess.CalledProcessError as exc:
            litConfig.warning(
                "unable to discover google-tests in %r: %s. Process output: %s"
                % (path, sys.exc_info()[1], exc.output))
            # This doesn't look like a valid gtest file.  This can
            # have a number of causes, none of them good.  For
            # instance, we could have created a broken executable.
            # Alternatively, someone has cruft in their test
            # directory.  If we don't return a test here, then no
            # failures will get reported, so return a dummy test name
            # so that the failure is reported later.
            yield 'failed_to_discover_tests_from_gtest'
            return

        nested_tests = []
        for ln in output.splitlines(False):  # Don't keep newlines.
            ln = lit.util.to_string(ln)

            if 'Running main() from gtest_main.cc' in ln:
                # Upstream googletest prints this to stdout prior to running
                # tests. LLVM removed that print statement in r61540, but we
                # handle it here in case upstream googletest is being used.
                continue

            # The test name list includes trailing comments beginning with
            # a '#' on some lines, so skip those. We don't support test names
            # that use escaping to embed '#' into their name as the names come
            # from C++ class and method names where such things are hard and
            # uninteresting to support.
            ln = ln.split('#', 1)[0].rstrip()
            if not ln.lstrip():
                continue

            index = 0
            while ln[index*2:index*2+2] == '  ':
                index += 1
            while len(nested_tests) > index:
                nested_tests.pop()

            ln = ln[index*2:]
            if ln.endswith('.'):
                nested_tests.append(ln)
            elif any([name.startswith('DISABLED_')
                      for name in nested_tests + [ln]]):
                # Gtest will internally skip these tests. No need to launch a
                # child process for it.
                continue
            else:
                yield ''.join(nested_tests) + ln

    def getTestsInDirectory(self, testSuite, path_in_suite,
                            litConfig, localConfig):
        source_path = testSuite.getSourcePath(path_in_suite)
        for subdir in self.test_sub_dirs:
            dir_path = os.path.join(source_path, subdir)
            if not os.path.isdir(dir_path):
                continue
            for fn in lit.util.listdir_files(dir_path,
                                             suffixes=self.test_suffixes):
                # Discover the tests in this executable.
                execpath = os.path.join(source_path, subdir, fn)
                testnames = self.getGTestTests(execpath, litConfig, localConfig)
                for testname in testnames:
                    testPath = path_in_suite + (subdir, fn, testname)
                    yield lit.Test.Test(testSuite, testPath, localConfig,
                                        file_path=execpath)

    def execute(self, test, litConfig):
        testPath,testName = os.path.split(test.getSourcePath())
        while not os.path.exists(testPath):
            # Handle GTest parametrized and typed tests, whose name includes
            # some '/'s.
            testPath, namePrefix = os.path.split(testPath)
            testName = namePrefix + '/' + testName

        cmd = [testPath, '--gtest_filter=' + testName]
        cmd = self.maybeAddPythonToCmd(cmd)
        if litConfig.useValgrind:
            cmd = litConfig.valgrindArgs + cmd

        if litConfig.noExecute:
            return lit.Test.PASS, ''

        try:
            out, err, exitCode = lit.util.executeCommand(
                cmd, env=test.config.environment,
                timeout=litConfig.maxIndividualTestTime)
        except lit.util.ExecuteCommandTimeoutException:
            return (lit.Test.TIMEOUT,
                    'Reached timeout of {} seconds'.format(
                        litConfig.maxIndividualTestTime)
                   )

        if exitCode:
            return lit.Test.FAIL, out + err

        passing_test_line = '[  PASSED  ] 1 test.'
        if passing_test_line not in out:
            msg = ('Unable to find %r in gtest output:\n\n%s%s' %
                   (passing_test_line, out, err))
            return lit.Test.UNRESOLVED, msg

        return lit.Test.PASS,''

    def maybeAddPythonToCmd(self, cmd):
        """Insert the python exe into the command if cmd[0] ends in .py

        We cannot rely on the system to interpret shebang lines for us on
        Windows, so add the python executable to the command if this is a .py
        script.
        """
        if cmd[0].endswith('.py'):
            return [sys.executable] + cmd
        return cmd
