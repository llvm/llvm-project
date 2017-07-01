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
        self.test_sub_dirs = os.path.normcase(str(test_sub_dirs)).split(';')
        self.test_suffix = str(test_suffix)

        # On Windows, assume tests will also end in '.exe'.
        if kIsWindows:
            self.test_suffix += '.exe'

    def getGTestTests(self, path, litConfig, localConfig):
        """getGTestTests(path) - [name]

        Return the tests available in gtest executable.

        Args:
          path: String path to a gtest executable
          litConfig: LitConfig instance
          localConfig: TestingConfig instance"""

        try:
            output = subprocess.check_output([path, '--gtest_list_tests'],
                                             env=localConfig.environment)
        except subprocess.CalledProcessError as exc:
            litConfig.warning(
                "unable to discover google-tests in %r: %s. Process output: %s"
                % (path, sys.exc_info()[1], exc.output))
            raise StopIteration

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
            for fn in lit.util.listdir_files(os.path.join(source_path, subdir),
                                             suffixes={self.test_suffix}):
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
