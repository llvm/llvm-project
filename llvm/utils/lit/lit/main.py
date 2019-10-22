#!/usr/bin/env python

"""
lit - LLVM Integrated Tester.

See lit.pod for more information.
"""

import os
import platform
import sys

import lit.cl_arguments
import lit.discovery
import lit.display
import lit.LitConfig
import lit.run
import lit.Test
import lit.util

def main(builtinParameters = {}):
    opts = lit.cl_arguments.parse_args()

    if opts.show_version:
        print("lit %s" % (lit.__version__,))
        return

    userParams = create_user_parameters(builtinParameters, opts)
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
        debug = opts.debug,
        isWindows = isWindows,
        params = userParams,
        config_prefix = opts.configPrefix,
        maxFailures = opts.maxFailures,
        echo_all_commands = opts.echoAllCommands)

    # Perform test discovery.
    tests = lit.discovery.find_tests_for_inputs(litConfig, opts.test_paths)

    # Command line overrides configuration for maxIndividualTestTime.
    if opts.maxIndividualTestTime is not None:  # `not None` is important (default: 0)
        if opts.maxIndividualTestTime != litConfig.maxIndividualTestTime:
            litConfig.note(('The test suite configuration requested an individual'
                ' test timeout of {0} seconds but a timeout of {1} seconds was'
                ' requested on the command line. Forcing timeout to be {1}'
                ' seconds')
                .format(litConfig.maxIndividualTestTime,
                        opts.maxIndividualTestTime))
            litConfig.maxIndividualTestTime = opts.maxIndividualTestTime

    if opts.showSuites or opts.showTests:
        print_suites_or_tests(tests, opts)
        return

    # Select and order the tests.
    numTotalTests = len(tests)

    if opts.filter:
        tests = [t for t in tests if opts.filter.search(t.getFullName())]

    order_tests(tests, opts)

    # Then optionally restrict our attention to a shard of the tests.
    if (opts.numShards is not None) or (opts.runShard is not None):
        num_tests = len(tests)
        # Note: user views tests and shard numbers counting from 1.
        test_ixs = range(opts.runShard - 1, num_tests, opts.numShards)
        tests = [tests[i] for i in test_ixs]
        # Generate a preview of the first few test indices in the shard
        # to accompany the arithmetic expression, for clarity.
        preview_len = 3
        ix_preview = ", ".join([str(i+1) for i in test_ixs[:preview_len]])
        if len(test_ixs) > preview_len:
            ix_preview += ", ..."
        litConfig.note('Selecting shard %d/%d = size %d/%d = tests #(%d*k)+%d = [%s]' %
                       (opts.runShard, opts.numShards,
                        len(tests), num_tests,
                        opts.numShards, opts.runShard, ix_preview))

    # Finally limit the number of tests, if desired.
    if opts.maxTests is not None:
        tests = tests[:opts.maxTests]

    # Don't create more workers than tests.
    opts.numWorkers = min(len(tests), opts.numWorkers)

    testing_time = run_tests(tests, litConfig, opts, numTotalTests)

    if not opts.quiet:
        print('Testing Time: %.2fs' % (testing_time,))

    print_summary(tests, opts)

    # Write out the test data, if requested.
    if opts.output_path:
        write_test_results(tests, litConfig, testing_time, opts.output_path)
    if opts.xunit_output_file:
        write_test_results_xunit(tests, opts)

    # If we encountered any additional errors, exit abnormally.
    if litConfig.numErrors:
        sys.stderr.write('\n%d error(s), exiting.\n' % litConfig.numErrors)
        sys.exit(2)

    # Warn about warnings.
    if litConfig.numWarnings:
        sys.stderr.write('\n%d warning(s) in tests.\n' % litConfig.numWarnings)

    has_failure = any(t.result.code.isFailure for t in tests)
    if has_failure:
        sys.exit(1)


def create_user_parameters(builtinParameters, opts):
    userParams = dict(builtinParameters)
    for entry in opts.userParameters:
        if '=' not in entry:
            name,val = entry,''
        else:
            name,val = entry.split('=', 1)
        userParams[name] = val
    return userParams

def print_suites_or_tests(tests, opts):
    # Aggregate the tests by suite.
    suitesAndTests = {}
    for result_test in tests:
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

def order_tests(tests, opts):
    if opts.shuffle:
        import random
        random.shuffle(tests)
    elif opts.incremental:
        tests.sort(key=by_mtime, reverse=True)
    else:
        tests.sort(key=lambda t: (not t.isEarlyTest(), t.getFullName()))

def by_mtime(test):
    fname = test.getFilePath()
    try:
        return os.path.getmtime(fname)
    except:
        return 0

def update_incremental_cache(test):
    if not test.result.code.isFailure:
        return
    fname = test.getFilePath()
    os.utime(fname, None)

def run_tests(tests, litConfig, opts, numTotalTests):
    display = lit.display.create_display(opts, len(tests), numTotalTests,
                                         opts.numWorkers)
    def progress_callback(test):
        display.update(test)
        if opts.incremental:
            update_incremental_cache(test)

    run = lit.run.create_run(tests, litConfig, opts.numWorkers,
                             progress_callback, opts.maxTime)

    try:
        elapsed = run_tests_in_tmp_dir(run.execute, litConfig)
    except KeyboardInterrupt:
        #TODO(yln): should we attempt to cleanup the progress bar here?
        sys.exit(2)
    # TODO(yln): display.finish_interrupted(), which shows the most recently started test
    # TODO(yln): change display to update when test starts, not when test completes
    # Ensure everything still works with SimpleProgressBar as well
    # finally:
    #     display.finish()

    display.finish()
    return elapsed

def run_tests_in_tmp_dir(run_callback, litConfig):
    # Create a temp directory inside the normal temp directory so that we can
    # try to avoid temporary test file leaks. The user can avoid this behavior
    # by setting LIT_PRESERVES_TMP in the environment, so they can easily use
    # their own temp directory to monitor temporary file leaks or handle them at
    # the buildbot level.
    tmp_dir = None
    if 'LIT_PRESERVES_TMP' not in os.environ:
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="lit_tmp_")
        os.environ.update({
                'TMPDIR': tmp_dir,
                'TMP': tmp_dir,
                'TEMP': tmp_dir,
                'TEMPDIR': tmp_dir,
                })
    # FIXME: If Python does not exit cleanly, this directory will not be cleaned
    # up. We should consider writing the lit pid into the temp directory,
    # scanning for stale temp directories, and deleting temp directories whose
    # lit process has died.
    try:
        return run_callback()
    finally:
        if tmp_dir:
            try:
                import shutil
                shutil.rmtree(tmp_dir)
            except:
                # FIXME: Re-try after timeout on Windows.
                litConfig.warning("Failed to delete temp directory '%s'" % tmp_dir)

def print_summary(tests, opts):
    byCode = {}
    for test in tests:
        if test.result.code not in byCode:
            byCode[test.result.code] = []
        byCode[test.result.code].append(test)

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

    if opts.timeTests and tests:
        # Order by time.
        test_times = [(test.getFullName(), test.result.elapsed)
                      for test in tests]
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

def write_test_results(tests, lit_config, testing_time, output_path):
    # Construct the data we will write.
    data = {}
    # Encode the current lit version as a schema version.
    data['__version__'] = lit.__versioninfo__
    data['elapsed'] = testing_time
    # FIXME: Record some information on the lit configuration used?
    # FIXME: Record information from the individual test suites?

    # Encode the tests.
    data['tests'] = tests_data = []
    for test in tests:
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
        import json
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')
    finally:
        f.close()

def write_test_results_xunit(tests, opts):
    from xml.sax.saxutils import quoteattr
    # Collect the tests, indexed by test suite
    by_suite = {}
    for result_test in tests:
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

if __name__=='__main__':
    main()
