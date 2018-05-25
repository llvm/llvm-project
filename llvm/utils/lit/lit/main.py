#!/usr/bin/env python

"""
lit - LLVM Integrated Tester.

See lit.pod for more information.
"""

from __future__ import absolute_import
import os
import platform
import random
import re
import sys
import time
import argparse
import tempfile
import shutil
from xml.sax.saxutils import quoteattr

import lit.ProgressBar
import lit.LitConfig
import lit.Test
import lit.run
import lit.util
import lit.discovery

class TestingProgressDisplay(object):
    def __init__(self, opts, numTests, progressBar=None):
        self.opts = opts
        self.numTests = numTests
        self.progressBar = progressBar
        self.completed = 0

    def finish(self):
        if self.progressBar:
            self.progressBar.clear()
        elif self.opts.quiet:
            pass
        elif self.opts.succinct:
            sys.stdout.write('\n')

    def update(self, test):
        self.completed += 1

        if self.opts.incremental:
            update_incremental_cache(test)

        if self.progressBar:
            self.progressBar.update(float(self.completed)/self.numTests,
                                    test.getFullName())

        shouldShow = test.result.code.isFailure or \
            self.opts.showAllOutput or \
            (not self.opts.quiet and not self.opts.succinct)
        if not shouldShow:
            return

        if self.progressBar:
            self.progressBar.clear()

        # Show the test result line.
        test_name = test.getFullName()
        print('%s: %s (%d of %d)' % (test.result.code.name, test_name,
                                     self.completed, self.numTests))

        # Show the test failure output, if requested.
        if (test.result.code.isFailure and self.opts.showOutput) or \
           self.opts.showAllOutput:
            if test.result.code.isFailure:
                print("%s TEST '%s' FAILED %s" % ('*'*20, test.getFullName(),
                                                  '*'*20))
            print(test.result.output)
            print("*" * 20)

        # Report test metrics, if present.
        if test.result.metrics:
            print("%s TEST '%s' RESULTS %s" % ('*'*10, test.getFullName(),
                                               '*'*10))
            items = sorted(test.result.metrics.items())
            for metric_name, value in items:
                print('%s: %s ' % (metric_name, value.format()))
            print("*" * 10)

        # Report micro-tests, if present
        if test.result.microResults:
            items = sorted(test.result.microResults.items())
            for micro_test_name, micro_test in items:
                print("%s MICRO-TEST: %s" %
                         ('*'*3, micro_test_name))
   
                if micro_test.metrics:
                    sorted_metrics = sorted(micro_test.metrics.items())
                    for metric_name, value in sorted_metrics:
                        print('    %s:  %s ' % (metric_name, value.format()))

        # Ensure the output is flushed.
        sys.stdout.flush()

def write_test_results(run, lit_config, testing_time, output_path):
    try:
        import json
    except ImportError:
        lit_config.fatal('test output unsupported with Python 2.5')

    # Construct the data we will write.
    data = {}
    # Encode the current lit version as a schema version.
    data['__version__'] = lit.__versioninfo__
    data['elapsed'] = testing_time
    # FIXME: Record some information on the lit configuration used?
    # FIXME: Record information from the individual test suites?

    # Encode the tests.
    data['tests'] = tests_data = []
    for test in run.tests:
        test_data = {
            'name' : test.getFullName(),
            'code' : test.result.code.name,
            'output' : test.result.output,
            'elapsed' : test.result.elapsed }

        # Add test metrics, if present.
        if test.result.metrics:
            test_data['metrics'] = metrics_data = {}
            for key, value in test.result.metrics.items():
                metrics_data[key] = value.todata()

        # Report micro-tests separately, if present
        if test.result.microResults:
            for key, micro_test in test.result.microResults.items():
                # Expand parent test name with micro test name
                parent_name = test.getFullName()
                micro_full_name = parent_name + ':' + key

                micro_test_data = {
                    'name' : micro_full_name,
                    'code' : micro_test.code.name,
                    'output' : micro_test.output,
                    'elapsed' : micro_test.elapsed }
                if micro_test.metrics:
                    micro_test_data['metrics'] = micro_metrics_data = {}
                    for key, value in micro_test.metrics.items():
                        micro_metrics_data[key] = value.todata()

                tests_data.append(micro_test_data)

        tests_data.append(test_data)

    # Write the output.
    f = open(output_path, 'w')
    try:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')
    finally:
        f.close()

def update_incremental_cache(test):
    if not test.result.code.isFailure:
        return
    fname = test.getFilePath()
    os.utime(fname, None)

def sort_by_incremental_cache(run):
    def sortIndex(test):
        fname = test.getFilePath()
        try:
            return -os.path.getmtime(fname)
        except:
            return 0
    run.tests.sort(key = lambda t: sortIndex(t))

def main(builtinParameters = {}):
    # Create a temp directory inside the normal temp directory so that we can
    # try to avoid temporary test file leaks. The user can avoid this behavior
    # by setting LIT_PRESERVES_TMP in the environment, so they can easily use
    # their own temp directory to monitor temporary file leaks or handle them at
    # the buildbot level.
    lit_tmp = None
    if 'LIT_PRESERVES_TMP' not in os.environ:
        lit_tmp = tempfile.mkdtemp(prefix="lit_tmp_")
        os.environ.update({
                'TMPDIR': lit_tmp,
                'TMP': lit_tmp,
                'TEMP': lit_tmp,
                'TEMPDIR': lit_tmp,
                })
    # FIXME: If Python does not exit cleanly, this directory will not be cleaned
    # up. We should consider writing the lit pid into the temp directory,
    # scanning for stale temp directories, and deleting temp directories whose
    # lit process has died.
    try:
        main_with_tmp(builtinParameters)
    finally:
        if lit_tmp:
            try:
                shutil.rmtree(lit_tmp)
            except:
                # FIXME: Re-try after timeout on Windows.
                pass

def main_with_tmp(builtinParameters):
    parser = argparse.ArgumentParser()
    parser.add_argument('test_paths',
                        nargs='*',
                        help='Files or paths to include in the test suite')

    parser.add_argument("--version", dest="show_version",
                      help="Show version and exit",
                      action="store_true", default=False)
    parser.add_argument("-j", "--threads", dest="numThreads", metavar="N",
                      help="Number of testing threads",
                      type=int, default=None)
    parser.add_argument("--config-prefix", dest="configPrefix",
                      metavar="NAME", help="Prefix for 'lit' config files",
                      action="store", default=None)
    parser.add_argument("-D", "--param", dest="userParameters",
                      metavar="NAME=VAL",
                      help="Add 'NAME' = 'VAL' to the user defined parameters",
                      type=str, action="append", default=[])

    format_group = parser.add_argument_group("Output Format")
    # FIXME: I find these names very confusing, although I like the
    # functionality.
    format_group.add_argument("-q", "--quiet",
                     help="Suppress no error output",
                     action="store_true", default=False)
    format_group.add_argument("-s", "--succinct",
                     help="Reduce amount of output",
                     action="store_true", default=False)
    format_group.add_argument("-v", "--verbose", dest="showOutput",
                     help="Show test output for failures",
                     action="store_true", default=False)
    format_group.add_argument("-vv", "--echo-all-commands",
                     dest="echoAllCommands",
                     action="store_true", default=False,
                     help="Echo all commands as they are executed to stdout.\
                     In case of failure, last command shown will be the\
                     failing one.")
    format_group.add_argument("-a", "--show-all", dest="showAllOutput",
                     help="Display all commandlines and output",
                     action="store_true", default=False)
    format_group.add_argument("-o", "--output", dest="output_path",
                     help="Write test results to the provided path",
                     action="store", metavar="PATH")
    format_group.add_argument("--no-progress-bar", dest="useProgressBar",
                     help="Do not use curses based progress bar",
                     action="store_false", default=True)
    format_group.add_argument("--show-unsupported",
                     help="Show unsupported tests",
                     action="store_true", default=False)
    format_group.add_argument("--show-xfail",
                     help="Show tests that were expected to fail",
                     action="store_true", default=False)

    execution_group = parser.add_argument_group("Test Execution")
    execution_group.add_argument("--path",
                     help="Additional paths to add to testing environment",
                     action="append", type=str, default=[])
    execution_group.add_argument("--vg", dest="useValgrind",
                     help="Run tests under valgrind",
                     action="store_true", default=False)
    execution_group.add_argument("--vg-leak", dest="valgrindLeakCheck",
                     help="Check for memory leaks under valgrind",
                     action="store_true", default=False)
    execution_group.add_argument("--vg-arg", dest="valgrindArgs", metavar="ARG",
                     help="Specify an extra argument for valgrind",
                     type=str, action="append", default=[])
    execution_group.add_argument("--time-tests", dest="timeTests",
                     help="Track elapsed wall time for each test",
                     action="store_true", default=False)
    execution_group.add_argument("--no-execute", dest="noExecute",
                     help="Don't execute any tests (assume PASS)",
                     action="store_true", default=False)
    execution_group.add_argument("--xunit-xml-output", dest="xunit_output_file",
                      help=("Write XUnit-compatible XML test reports to the"
                            " specified file"), default=None)
    execution_group.add_argument("--timeout", dest="maxIndividualTestTime",
                     help="Maximum time to spend running a single test (in seconds)."
                     "0 means no time limit. [Default: 0]",
                    type=int, default=None)
    execution_group.add_argument("--max-failures", dest="maxFailures",
                     help="Stop execution after the given number of failures.",
                     action="store", type=int, default=None)

    selection_group = parser.add_argument_group("Test Selection")
    selection_group.add_argument("--max-tests", dest="maxTests", metavar="N",
                     help="Maximum number of tests to run",
                     action="store", type=int, default=None)
    selection_group.add_argument("--max-time", dest="maxTime", metavar="N",
                     help="Maximum time to spend testing (in seconds)",
                     action="store", type=float, default=None)
    selection_group.add_argument("--shuffle",
                     help="Run tests in random order",
                     action="store_true", default=False)
    selection_group.add_argument("-i", "--incremental",
                     help="Run modified and failing tests first (updates "
                     "mtimes)",
                     action="store_true", default=False)
    selection_group.add_argument("--filter", metavar="REGEX",
                     help=("Only run tests with paths matching the given "
                           "regular expression"),
                     action="store",
                     default=os.environ.get("LIT_FILTER"))
    selection_group.add_argument("--num-shards", dest="numShards", metavar="M",
                     help="Split testsuite into M pieces and only run one",
                     action="store", type=int,
                     default=os.environ.get("LIT_NUM_SHARDS"))
    selection_group.add_argument("--run-shard", dest="runShard", metavar="N",
                     help="Run shard #N of the testsuite",
                     action="store", type=int,
                     default=os.environ.get("LIT_RUN_SHARD"))

    debug_group = parser.add_argument_group("Debug and Experimental Options")
    debug_group.add_argument("--debug",
                      help="Enable debugging (for 'lit' development)",
                      action="store_true", default=False)
    debug_group.add_argument("--show-suites", dest="showSuites",
                      help="Show discovered test suites",
                      action="store_true", default=False)
    debug_group.add_argument("--show-tests", dest="showTests",
                      help="Show all discovered tests",
                      action="store_true", default=False)
    debug_group.add_argument("--single-process", dest="singleProcess",
                      help="Don't run tests in parallel.  Intended for debugging "
                      "single test failures",
                      action="store_true", default=False)

    opts = parser.parse_args()
    args = opts.test_paths

    if opts.show_version:
        print("lit %s" % (lit.__version__,))
        return

    if not args:
        parser.error('No inputs specified')

    if opts.numThreads is None:
        opts.numThreads = lit.util.detectCPUs()

    if opts.maxFailures == 0:
        parser.error("Setting --max-failures to 0 does not have any effect.")

    if opts.echoAllCommands:
        opts.showOutput = True

    inputs = args

    # Create the user defined parameters.
    userParams = dict(builtinParameters)
    for entry in opts.userParameters:
        if '=' not in entry:
            name,val = entry,''
        else:
            name,val = entry.split('=', 1)
        userParams[name] = val

    # Decide what the requested maximum indvidual test time should be
    if opts.maxIndividualTestTime is not None:
        maxIndividualTestTime = opts.maxIndividualTestTime
    else:
        # Default is zero
        maxIndividualTestTime = 0

    isWindows = platform.system() == 'Windows'

    # Create the global config object.
    litConfig = lit.LitConfig.LitConfig(
        progname = os.path.basename(sys.argv[0]),
        path = opts.path,
        quiet = opts.quiet,
        useValgrind = opts.useValgrind,
        valgrindLeakCheck = opts.valgrindLeakCheck,
        valgrindArgs = opts.valgrindArgs,
        noExecute = opts.noExecute,
        singleProcess = opts.singleProcess,
        debug = opts.debug,
        isWindows = isWindows,
        params = userParams,
        config_prefix = opts.configPrefix,
        maxIndividualTestTime = maxIndividualTestTime,
        maxFailures = opts.maxFailures,
        parallelism_groups = {},
        echo_all_commands = opts.echoAllCommands)

    # Perform test discovery.
    run = lit.run.Run(litConfig,
                      lit.discovery.find_tests_for_inputs(litConfig, inputs))

    # After test discovery the configuration might have changed
    # the maxIndividualTestTime. If we explicitly set this on the
    # command line then override what was set in the test configuration
    if opts.maxIndividualTestTime is not None:
        if opts.maxIndividualTestTime != litConfig.maxIndividualTestTime:
            litConfig.note(('The test suite configuration requested an individual'
                ' test timeout of {0} seconds but a timeout of {1} seconds was'
                ' requested on the command line. Forcing timeout to be {1}'
                ' seconds')
                .format(litConfig.maxIndividualTestTime,
                        opts.maxIndividualTestTime))
            litConfig.maxIndividualTestTime = opts.maxIndividualTestTime

    if opts.showSuites or opts.showTests:
        # Aggregate the tests by suite.
        suitesAndTests = {}
        for result_test in run.tests:
            if result_test.suite not in suitesAndTests:
                suitesAndTests[result_test.suite] = []
            suitesAndTests[result_test.suite].append(result_test)
        suitesAndTests = list(suitesAndTests.items())
        suitesAndTests.sort(key = lambda item: item[0].name)

        # Show the suites, if requested.
        if opts.showSuites:
            print('-- Test Suites --')
            for ts,ts_tests in suitesAndTests:
                print('  %s - %d tests' %(ts.name, len(ts_tests)))
                print('    Source Root: %s' % ts.source_root)
                print('    Exec Root  : %s' % ts.exec_root)
                if ts.config.available_features:
                    print('    Available Features : %s' % ' '.join(
                        sorted(ts.config.available_features)))

        # Show the tests, if requested.
        if opts.showTests:
            print('-- Available Tests --')
            for ts,ts_tests in suitesAndTests:
                ts_tests.sort(key = lambda test: test.path_in_suite)
                for test in ts_tests:
                    print('  %s' % (test.getFullName(),))

        # Exit.
        sys.exit(0)

    # Select and order the tests.
    numTotalTests = len(run.tests)

    # First, select based on the filter expression if given.
    if opts.filter:
        try:
            rex = re.compile(opts.filter)
        except:
            parser.error("invalid regular expression for --filter: %r" % (
                    opts.filter))
        run.tests = [result_test for result_test in run.tests
                     if rex.search(result_test.getFullName())]

    # Then select the order.
    if opts.shuffle:
        random.shuffle(run.tests)
    elif opts.incremental:
        sort_by_incremental_cache(run)
    else:
        run.tests.sort(key = lambda t: (not t.isEarlyTest(), t.getFullName()))

    # Then optionally restrict our attention to a shard of the tests.
    if (opts.numShards is not None) or (opts.runShard is not None):
        if (opts.numShards is None) or (opts.runShard is None):
            parser.error("--num-shards and --run-shard must be used together")
        if opts.numShards <= 0:
            parser.error("--num-shards must be positive")
        if (opts.runShard < 1) or (opts.runShard > opts.numShards):
            parser.error("--run-shard must be between 1 and --num-shards (inclusive)")
        num_tests = len(run.tests)
        # Note: user views tests and shard numbers counting from 1.
        test_ixs = range(opts.runShard - 1, num_tests, opts.numShards)
        run.tests = [run.tests[i] for i in test_ixs]
        # Generate a preview of the first few test indices in the shard
        # to accompany the arithmetic expression, for clarity.
        preview_len = 3
        ix_preview = ", ".join([str(i+1) for i in test_ixs[:preview_len]])
        if len(test_ixs) > preview_len:
            ix_preview += ", ..."
        litConfig.note('Selecting shard %d/%d = size %d/%d = tests #(%d*k)+%d = [%s]' %
                       (opts.runShard, opts.numShards,
                        len(run.tests), num_tests,
                        opts.numShards, opts.runShard, ix_preview))

    # Finally limit the number of tests, if desired.
    if opts.maxTests is not None:
        run.tests = run.tests[:opts.maxTests]

    # Don't create more threads than tests.
    opts.numThreads = min(len(run.tests), opts.numThreads)

    # Because some tests use threads internally, and at least on Linux each
    # of these threads counts toward the current process limit, try to
    # raise the (soft) process limit so that tests don't fail due to
    # resource exhaustion.
    try:
        cpus = lit.util.detectCPUs()
        desired_limit = opts.numThreads * cpus * 2 # the 2 is a safety factor

        # Import the resource module here inside this try block because it
        # will likely fail on Windows.
        import resource

        max_procs_soft, max_procs_hard = resource.getrlimit(resource.RLIMIT_NPROC)
        desired_limit = min(desired_limit, max_procs_hard)

        if max_procs_soft < desired_limit:
            resource.setrlimit(resource.RLIMIT_NPROC, (desired_limit, max_procs_hard))
            litConfig.note('raised the process limit from %d to %d' % \
                               (max_procs_soft, desired_limit))
    except:
        pass

    extra = ''
    if len(run.tests) != numTotalTests:
        extra = ' of %d' % numTotalTests
    header = '-- Testing: %d%s tests, %d threads --'%(len(run.tests), extra,
                                                      opts.numThreads)
    progressBar = None
    if not opts.quiet:
        if opts.succinct and opts.useProgressBar:
            try:
                tc = lit.ProgressBar.TerminalController()
                progressBar = lit.ProgressBar.ProgressBar(tc, header)
            except ValueError:
                print(header)
                progressBar = lit.ProgressBar.SimpleProgressBar('Testing: ')
        else:
            print(header)

    startTime = time.time()
    display = TestingProgressDisplay(opts, len(run.tests), progressBar)
    try:
        run.execute_tests(display, opts.numThreads, opts.maxTime)
    except KeyboardInterrupt:
        sys.exit(2)
    display.finish()

    testing_time = time.time() - startTime
    if not opts.quiet:
        print('Testing Time: %.2fs' % (testing_time,))

    # Write out the test data, if requested.
    if opts.output_path is not None:
        write_test_results(run, litConfig, testing_time, opts.output_path)

    # List test results organized by kind.
    hasFailures = False
    byCode = {}
    for test in run.tests:
        if test.result.code not in byCode:
            byCode[test.result.code] = []
        byCode[test.result.code].append(test)
        if test.result.code.isFailure:
            hasFailures = True

    # Print each test in any of the failing groups.
    for title,code in (('Unexpected Passing Tests', lit.Test.XPASS),
                       ('Failing Tests', lit.Test.FAIL),
                       ('Unresolved Tests', lit.Test.UNRESOLVED),
                       ('Unsupported Tests', lit.Test.UNSUPPORTED),
                       ('Expected Failing Tests', lit.Test.XFAIL),
                       ('Timed Out Tests', lit.Test.TIMEOUT)):
        if (lit.Test.XFAIL == code and not opts.show_xfail) or \
           (lit.Test.UNSUPPORTED == code and not opts.show_unsupported) or \
           (lit.Test.UNRESOLVED == code and (opts.maxFailures is not None)):
            continue
        elts = byCode.get(code)
        if not elts:
            continue
        print('*'*20)
        print('%s (%d):' % (title, len(elts)))
        for test in elts:
            print('    %s' % test.getFullName())
        sys.stdout.write('\n')

    if opts.timeTests and run.tests:
        # Order by time.
        test_times = [(test.getFullName(), test.result.elapsed)
                      for test in run.tests]
        lit.util.printHistogram(test_times, title='Tests')

    for name,code in (('Expected Passes    ', lit.Test.PASS),
                      ('Passes With Retry  ', lit.Test.FLAKYPASS),
                      ('Expected Failures  ', lit.Test.XFAIL),
                      ('Unsupported Tests  ', lit.Test.UNSUPPORTED),
                      ('Unresolved Tests   ', lit.Test.UNRESOLVED),
                      ('Unexpected Passes  ', lit.Test.XPASS),
                      ('Unexpected Failures', lit.Test.FAIL),
                      ('Individual Timeouts', lit.Test.TIMEOUT)):
        if opts.quiet and not code.isFailure:
            continue
        N = len(byCode.get(code,[]))
        if N:
            print('  %s: %d' % (name,N))

    if opts.xunit_output_file:
        # Collect the tests, indexed by test suite
        by_suite = {}
        for result_test in run.tests:
            suite = result_test.suite.config.name
            if suite not in by_suite:
                by_suite[suite] = {
                                   'passes'   : 0,
                                   'failures' : 0,
                                   'skipped': 0,
                                   'tests'    : [] }
            by_suite[suite]['tests'].append(result_test)
            if result_test.result.code.isFailure:
                by_suite[suite]['failures'] += 1
            elif result_test.result.code == lit.Test.UNSUPPORTED:
                by_suite[suite]['skipped'] += 1
            else:
                by_suite[suite]['passes'] += 1
        xunit_output_file = open(opts.xunit_output_file, "w")
        xunit_output_file.write("<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n")
        xunit_output_file.write("<testsuites>\n")
        for suite_name, suite in by_suite.items():
            safe_suite_name = quoteattr(suite_name.replace(".", "-"))
            xunit_output_file.write("<testsuite name=" + safe_suite_name)
            xunit_output_file.write(" tests=\"" + str(suite['passes'] +
              suite['failures'] + suite['skipped']) + "\"")
            xunit_output_file.write(" failures=\"" + str(suite['failures']) + "\"")
            xunit_output_file.write(" skipped=\"" + str(suite['skipped']) +
              "\">\n")

            for result_test in suite['tests']:
                result_test.writeJUnitXML(xunit_output_file)
                xunit_output_file.write("\n")
            xunit_output_file.write("</testsuite>\n")
        xunit_output_file.write("</testsuites>")
        xunit_output_file.close()

    # If we encountered any additional errors, exit abnormally.
    if litConfig.numErrors:
        sys.stderr.write('\n%d error(s), exiting.\n' % litConfig.numErrors)
        sys.exit(2)

    # Warn about warnings.
    if litConfig.numWarnings:
        sys.stderr.write('\n%d warning(s) in tests.\n' % litConfig.numWarnings)

    if hasFailures:
        sys.exit(1)
    sys.exit(0)

if __name__=='__main__':
    main()
