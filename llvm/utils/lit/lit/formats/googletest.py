from __future__ import absolute_import
import json
import math
import os
import shlex
import sys

import lit.Test
import lit.TestRunner
import lit.util
from .base import TestFormat

kIsWindows = sys.platform in ['win32', 'cygwin']

class GoogleTest(TestFormat):
    def __init__(self, test_sub_dirs, test_suffix, run_under = []):
        self.test_sub_dirs = str(test_sub_dirs).split(';')

        # On Windows, assume tests will also end in '.exe'.
        exe_suffix = str(test_suffix)
        if kIsWindows:
            exe_suffix += '.exe'

        # Also check for .py files for testing purposes.
        self.test_suffixes = {exe_suffix, test_suffix + '.py'}
        self.run_under = run_under

    def get_num_tests(self, path, localConfig):
        cmd = [path, '--gtest_list_tests', '--gtest_filter=-*DISABLED_*']
        if cmd[0].endswith('.py'):
            cmd = [sys.executable] + cmd
        out, _, exitCode = lit.util.executeCommand(cmd, env=localConfig.environment)
        if exitCode == 0:
            return sum(map(lambda line: line.startswith('  '), out.splitlines()))
        return None

    def getTestsInDirectory(self, testSuite, path_in_suite,
                            litConfig, localConfig):
        init_shard_size = 512 # number of tests in a shard
        core_count = lit.util.usable_core_count()
        source_path = testSuite.getSourcePath(path_in_suite)
        for subdir in self.test_sub_dirs:
            dir_path = os.path.join(source_path, subdir)
            if not os.path.isdir(dir_path):
                continue
            for fn in lit.util.listdir_files(dir_path,
                                             suffixes=self.test_suffixes):
                # Discover the tests in this executable.
                execpath = os.path.join(source_path, subdir, fn)
                num_tests = self.get_num_tests(execpath, localConfig)
                if num_tests is not None:
                    # Compute the number of shards.
                    shard_size = init_shard_size
                    nshard = int(math.ceil(num_tests/shard_size))
                    while nshard < core_count and shard_size > 1:
                        shard_size = shard_size//2
                        nshard = int(math.ceil(num_tests/shard_size))

                    # Create one lit test for each shard.
                    for idx in range(nshard):
                        testPath = path_in_suite + (subdir, fn,
                                                        str(idx), str(nshard))
                        json_file = '-'.join([execpath, testSuite.config.name,
                                              str(os.getpid()), str(idx),
                                              str(nshard)]) + '.json'
                        yield lit.Test.Test(testSuite, testPath, localConfig,
                                            file_path=execpath,
                                            gtest_json_file=json_file)
                else:
                    # This doesn't look like a valid gtest file.  This can
                    # have a number of causes, none of them good.  For
                    # instance, we could have created a broken executable.
                    # Alternatively, someone has cruft in their test
                    # directory.  If we don't return a test here, then no
                    # failures will get reported, so return a dummy test name
                    # so that the failure is reported later.
                    testPath = path_in_suite + (subdir, fn, 'failed_to_discover_tests_from_gtest')
                    yield lit.Test.Test(testSuite, testPath, localConfig, file_path=execpath)

    def execute(self, test, litConfig):
        if test.gtest_json_file is None:
            return lit.Test.FAIL, ''

        testPath,testName = os.path.split(test.getSourcePath())
        while not os.path.exists(testPath):
            # Handle GTest parametrized and typed tests, whose name includes
            # some '/'s.
            testPath, namePrefix = os.path.split(testPath)
            testName = namePrefix + '/' + testName

        testName,total_shards = os.path.split(testName)
        testName,shard_idx = os.path.split(testName)
        shard_env = {'GTEST_COLOR':'no','GTEST_TOTAL_SHARDS':total_shards, 'GTEST_SHARD_INDEX':shard_idx, 'GTEST_OUTPUT':'json:'+test.gtest_json_file}
        test.config.environment.update(shard_env)

        cmd = [testPath]
        cmd = self.prepareCmd(cmd)
        if litConfig.useValgrind:
            cmd = litConfig.valgrindArgs + cmd

        if litConfig.noExecute:
            return lit.Test.PASS, ''

        shard_envs= '\n'.join([k + '=' + v for k, v in shard_env.items()])
        shard_header = f"Script(shard):\n--\n{shard_envs}\n{' '.join(cmd)}\n--\n"

        try:
            _, _, exitCode = lit.util.executeCommand(
                cmd, env=test.config.environment,
                timeout=litConfig.maxIndividualTestTime)
        except lit.util.ExecuteCommandTimeoutException:
            return (lit.Test.TIMEOUT,
                    f'{shard_header}Reached timeout of '
                    f'{litConfig.maxIndividualTestTime} seconds')

        if not os.path.exists(test.gtest_json_file):
            errmsg = f"shard JSON output does not exist: %s" % (test.gtest_json_file)
            return lit.Test.FAIL, shard_header + errmsg

        if exitCode:
            output = shard_header + '\n'
            with open(test.gtest_json_file, encoding='utf-8') as f:
                testsuites = json.load(f)['testsuites']
                for testcase in testsuites:
                    for testinfo in testcase['testsuite']:
                        if testinfo['result'] == 'SUPPRESSED' or testinfo['result'] == 'SKIPPED':
                            continue
                        testname = testcase['name'] + '.' + testinfo['name']
                        header = f"Script:\n--\n{' '.join(cmd)} --gtest_filter={testname}\n--\n"
                        if 'failures' in testinfo:
                            output += header
                            for fail in testinfo['failures']:
                                output += fail['failure'] + '\n'
                            output += '\n'
                        elif testinfo['result'] != 'COMPLETED':
                            output += header
                            output += 'unresolved test result\n'
            return lit.Test.FAIL, output
        else:
            return lit.Test.PASS, ''

    def prepareCmd(self, cmd):
        """Insert interpreter if needed.

        It inserts the python exe into the command if cmd[0] ends in .py or caller
        specified run_under.
        We cannot rely on the system to interpret shebang lines for us on
        Windows, so add the python executable to the command if this is a .py
        script.
        """
        if cmd[0].endswith('.py'):
            cmd = [sys.executable] + cmd
        if self.run_under:
            if isinstance(self.run_under, list):
                cmd = self.run_under + cmd
            else:
                cmd = shlex.split(self.run_under) + cmd
        return cmd

    @staticmethod
    def post_process_shard_results(selected_tests, discovered_tests):
        def remove_gtest(tests):
            idxs = []
            for idx, t in enumerate(tests):
                if t.gtest_json_file:
                    idxs.append(idx)
            for i in range(len(idxs)):
                del tests[idxs[i]-i]

        remove_gtest(discovered_tests)
        gtests = [t for t in selected_tests if t.gtest_json_file]
        remove_gtest(selected_tests)
        for test in gtests:
            # In case gtest has bugs such that no JSON file was emitted.
            if not os.path.exists(test.gtest_json_file):
                selected_tests.append(test)
                discovered_tests.append(test)
                continue

            # Load json file to retrieve results.
            with open(test.gtest_json_file, encoding='utf-8') as f:
                testsuites = json.load(f)['testsuites']
                for testcase in testsuites:
                    for testinfo in testcase['testsuite']:
                        # Ignore disabled tests.
                        if testinfo['result'] == 'SUPPRESSED':
                            continue

                        testPath = test.path_in_suite[:-2] + (testcase['name'], testinfo['name'])
                        subtest = lit.Test.Test(test.suite, testPath,
                                                test.config, test.file_path)

                        testname = testcase['name'] + '.' + testinfo['name']
                        header = f"Script:\n--\n{test.file_path} --gtest_filter={testname}\n--\n"

                        output = ''
                        if testinfo['result'] == 'SKIPPED':
                            returnCode = lit.Test.SKIPPED
                        elif 'failures' in testinfo:
                            returnCode = lit.Test.FAIL
                            output = header
                            for fail in testinfo['failures']:
                                output += fail['failure'] + '\n'
                        elif testinfo['result'] == 'COMPLETED':
                            returnCode = lit.Test.PASS
                        else:
                            returnCode = lit.Test.UNRESOLVED
                            output = header + 'unresolved test result\n'

                        subtest.setResult(lit.Test.Result(returnCode, output, float(testinfo['time'][:-1])))

                        selected_tests.append(subtest)
                        discovered_tests.append(subtest)
            os.remove(test.gtest_json_file)

        return selected_tests, discovered_tests
