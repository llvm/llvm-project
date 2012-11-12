import os
import sys

import Test
import TestRunner
import Util

kIsWindows = sys.platform in ['win32', 'cygwin']

class GoogleTest(object):
    def __init__(self, test_sub_dir, test_suffix):
        self.test_sub_dir = os.path.normcase(str(test_sub_dir)).split(';')
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
            lines = Util.capture([path, '--gtest_list_tests'],
                                 env=localConfig.environment)
            if kIsWindows:
              lines = lines.replace('\r', '')
            lines = lines.split('\n')
        except:
            litConfig.error("unable to discover google-tests in %r" % path)
            raise StopIteration

        nested_tests = []
        for ln in lines:
            if not ln.strip():
                continue

            prefix = ''
            index = 0
            while ln[index*2:index*2+2] == '  ':
                index += 1
            while len(nested_tests) > index:
                nested_tests.pop()

            ln = ln[index*2:]
            if ln.endswith('.'):
                nested_tests.append(ln)
            else:
                yield ''.join(nested_tests) + ln

    def getTestsInDirectory(self, testSuite, path_in_suite,
                            litConfig, localConfig):
        source_path = testSuite.getSourcePath(path_in_suite)
        for filename in os.listdir(source_path):
            # Check for the one subdirectory (build directory) tests will be in.
            if not '.' in self.test_sub_dir:
                if not os.path.normcase(filename) in self.test_sub_dir:
                    continue

            filepath = os.path.join(source_path, filename)
            if not os.path.isdir(filepath):
                continue

            for subfilename in os.listdir(filepath):
                if subfilename.endswith(self.test_suffix):
                    execpath = os.path.join(filepath, subfilename)

                    # Discover the tests in this executable.
                    for name in self.getGTestTests(execpath, litConfig,
                                                   localConfig):
                        testPath = path_in_suite + (filename, subfilename, name)
                        yield Test.Test(testSuite, testPath, localConfig)

    def execute(self, test, litConfig):
        testPath,testName = os.path.split(test.getSourcePath())
        while not os.path.exists(testPath):
            # Handle GTest parametrized and typed tests, whose name includes
            # some '/'s.
            testPath, namePrefix = os.path.split(testPath)
            testName = os.path.join(namePrefix, testName)

        cmd = [testPath, '--gtest_filter=' + testName]
        if litConfig.useValgrind:
            cmd = litConfig.valgrindArgs + cmd

        out, err, exitCode = TestRunner.executeCommand(
            cmd, env=test.config.environment)

        if not exitCode:
            return Test.PASS,''

        return Test.FAIL, out + err

###

class FileBasedTest(object):
    def getTestsInDirectory(self, testSuite, path_in_suite,
                            litConfig, localConfig):
        source_path = testSuite.getSourcePath(path_in_suite)
        for filename in os.listdir(source_path):
            # Ignore dot files and excluded tests.
            if (filename.startswith('.') or
                filename in localConfig.excludes):
                continue

            filepath = os.path.join(source_path, filename)
            if not os.path.isdir(filepath):
                base,ext = os.path.splitext(filename)
                if ext in localConfig.suffixes:
                    yield Test.Test(testSuite, path_in_suite + (filename,),
                                    localConfig)

class ShTest(FileBasedTest):
    def __init__(self, execute_external = False):
        self.execute_external = execute_external

    def execute(self, test, litConfig):
        return TestRunner.executeShTest(test, litConfig,
                                        self.execute_external)

class TclTest(FileBasedTest):
    def __init__(self, ignoreStdErr=False):
        self.ignoreStdErr = ignoreStdErr
        
    def execute(self, test, litConfig):
        litConfig.ignoreStdErr = self.ignoreStdErr
        return TestRunner.executeTclTest(test, litConfig)

###

import re
import tempfile

class OneCommandPerFileTest:
    # FIXME: Refactor into generic test for running some command on a directory
    # of inputs.

    def __init__(self, command, dir, recursive=False,
                 pattern=".*", useTempInput=False):
        if isinstance(command, str):
            self.command = [command]
        else:
            self.command = list(command)
        if dir is not None:
            dir = str(dir)
        self.dir = dir
        self.recursive = bool(recursive)
        self.pattern = re.compile(pattern)
        self.useTempInput = useTempInput

    def getTestsInDirectory(self, testSuite, path_in_suite,
                            litConfig, localConfig):
        dir = self.dir
        if dir is None:
            dir = testSuite.getSourcePath(path_in_suite)

        for dirname,subdirs,filenames in os.walk(dir):
            if not self.recursive:
                subdirs[:] = []

            subdirs[:] = [d for d in subdirs
                          if (d != '.svn' and
                              d not in localConfig.excludes)]

            for filename in filenames:
                if (filename.startswith('.') or
                    not self.pattern.match(filename) or
                    filename in localConfig.excludes):
                    continue

                path = os.path.join(dirname,filename)
                suffix = path[len(dir):]
                if suffix.startswith(os.sep):
                    suffix = suffix[1:]
                test = Test.Test(testSuite,
                                 path_in_suite + tuple(suffix.split(os.sep)),
                                 localConfig)
                # FIXME: Hack?
                test.source_path = path
                yield test

    def createTempInput(self, tmp, test):
        abstract

    def execute(self, test, litConfig):
        if test.config.unsupported:
            return (Test.UNSUPPORTED, 'Test is unsupported')

        cmd = list(self.command)

        # If using temp input, create a temporary file and hand it to the
        # subclass.
        if self.useTempInput:
            tmp = tempfile.NamedTemporaryFile(suffix='.cpp')
            self.createTempInput(tmp, test)
            tmp.flush()
            cmd.append(tmp.name)
        elif hasattr(test, 'source_path'):
            cmd.append(test.source_path)
        else:
            cmd.append(test.getSourcePath())

        out, err, exitCode = TestRunner.executeCommand(cmd)

        diags = out + err
        if not exitCode and not diags.strip():
            return Test.PASS,''

        # Try to include some useful information.
        report = """Command: %s\n""" % ' '.join(["'%s'" % a
                                                 for a in cmd])
        if self.useTempInput:
            report += """Temporary File: %s\n""" % tmp.name
            report += "--\n%s--\n""" % open(tmp.name).read()
        report += """Output:\n--\n%s--""" % diags

        return Test.FAIL, report

class SyntaxCheckTest(OneCommandPerFileTest):
    def __init__(self, compiler, dir, extra_cxx_args=[], *args, **kwargs):
        cmd = [compiler, '-x', 'c++', '-fsyntax-only'] + extra_cxx_args
        OneCommandPerFileTest.__init__(self, cmd, dir,
                                       useTempInput=1, *args, **kwargs)

    def createTempInput(self, tmp, test):
        print >>tmp, '#include "%s"' % test.source_path
