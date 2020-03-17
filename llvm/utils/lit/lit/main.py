"""
lit - LLVM Integrated Tester.

See lit.pod for more information.
"""

import os
import platform
import sys
import time

import lit.cl_arguments
import lit.discovery
import lit.display
import lit.LitConfig
import lit.run
import lit.Test
import lit.util


def main(builtin_params={}):
    opts = lit.cl_arguments.parse_args()

    params = create_params(builtin_params, opts.user_params)
    is_windows = platform.system() == 'Windows'

    lit_config = lit.LitConfig.LitConfig(
        progname=os.path.basename(sys.argv[0]),
        path=opts.path,
        quiet=opts.quiet,
        useValgrind=opts.useValgrind,
        valgrindLeakCheck=opts.valgrindLeakCheck,
        valgrindArgs=opts.valgrindArgs,
        noExecute=opts.noExecute,
        debug=opts.debug,
        isWindows=is_windows,
        params=params,
        config_prefix=opts.configPrefix,
        echo_all_commands=opts.echoAllCommands)

    discovered_tests = lit.discovery.find_tests_for_inputs(lit_config, opts.test_paths)
    if not discovered_tests:
        sys.stderr.write('error: did not discover any tests for provided path(s)\n')
        sys.exit(2)

    # Command line overrides configuration for maxIndividualTestTime.
    if opts.maxIndividualTestTime is not None:  # `not None` is important (default: 0)
        if opts.maxIndividualTestTime != lit_config.maxIndividualTestTime:
            lit_config.note(('The test suite configuration requested an individual'
                ' test timeout of {0} seconds but a timeout of {1} seconds was'
                ' requested on the command line. Forcing timeout to be {1}'
                ' seconds')
                .format(lit_config.maxIndividualTestTime,
                        opts.maxIndividualTestTime))
            lit_config.maxIndividualTestTime = opts.maxIndividualTestTime

    if opts.showSuites or opts.showTests:
        print_suites_or_tests(discovered_tests, opts)
        return

    if opts.filter:
        filtered_tests = [t for t in discovered_tests if
                          opts.filter.search(t.getFullName())]
        if not filtered_tests:
            sys.stderr.write('error: filter did not match any tests '
                             '(of %d discovered).  ' % len(discovered_tests))
            if opts.allow_empty_runs:
                sys.stderr.write('Suppressing error because '
                                 "'--allow-empty-runs' was specified.\n")
                sys.exit(0)
            else:
                sys.stderr.write("Use '--allow-empty-runs' to suppress this "
                                 'error.\n')
                sys.exit(2)
    else:
        filtered_tests = discovered_tests

    determine_order(filtered_tests, opts.order)

    if opts.shard:
        (run, shards) = opts.shard
        filtered_tests = filter_by_shard(filtered_tests, run, shards, lit_config)
        if not filtered_tests:
            sys.stderr.write('warning: shard does not contain any tests.  '
                             'Consider decreasing the number of shards.\n')
            sys.exit(0)

    if opts.max_tests:
        filtered_tests = filtered_tests[:opts.max_tests]

    opts.workers = min(len(filtered_tests), opts.workers)

    start = time.time()
    run_tests(filtered_tests, lit_config, opts, len(discovered_tests))
    elapsed = time.time() - start

    executed_tests = [t for t in filtered_tests if t.result]

    print_summary(executed_tests, elapsed, opts)

    if opts.output_path:
        #TODO(yln): pass in discovered_tests
        write_test_results(executed_tests, lit_config, elapsed, opts.output_path)
    if opts.xunit_output_file:
        write_test_results_xunit(executed_tests, opts)

    if lit_config.numErrors:
        sys.stderr.write('\n%d error(s) in tests\n' % lit_config.numErrors)
        sys.exit(2)

    if lit_config.numWarnings:
        sys.stderr.write('\n%d warning(s) in tests\n' % lit_config.numWarnings)

    has_failure = any(t.isFailure() for t in executed_tests)
    if has_failure:
        sys.exit(1)


def create_params(builtin_params, user_params):
    def parse(p):
        return p.split('=', 1) if '=' in p else (p, '')

    params = dict(builtin_params)
    params.update([parse(p) for p in user_params])
    return params

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


def determine_order(tests, order):
    assert order in ['default', 'random', 'failing-first']
    if order == 'default':
        tests.sort(key=lambda t: (not t.isEarlyTest(), t.getFullName()))
    elif order == 'random':
        import random
        random.shuffle(tests)
    else:
        def by_mtime(test):
            return os.path.getmtime(test.getFilePath())
        tests.sort(key=by_mtime, reverse=True)


def touch_file(test):
    if test.isFailure():
        os.utime(test.getFilePath(), None)


def filter_by_shard(tests, run, shards, lit_config):
    test_ixs = range(run - 1, len(tests), shards)
    selected_tests = [tests[i] for i in test_ixs]

    # For clarity, generate a preview of the first few test indices in the shard
    # to accompany the arithmetic expression.
    preview_len = 3
    preview = ", ".join([str(i + 1) for i in test_ixs[:preview_len]])
    if len(test_ixs) > preview_len:
        preview += ", ..."
    # TODO(python3): string interpolation
    msg = 'Selecting shard {run}/{shards} = size {sel_tests}/{total_tests} = ' \
          'tests #({shards}*k)+{run} = [{preview}]'.format(
              run=run, shards=shards, sel_tests=len(selected_tests),
              total_tests=len(tests), preview=preview)
    lit_config.note(msg)
    return selected_tests


def run_tests(tests, lit_config, opts, numTotalTests):
    display = lit.display.create_display(opts, len(tests), numTotalTests,
                                         opts.workers)
    def progress_callback(test):
        display.update(test)
        if opts.order == 'failing-first':
            touch_file(test)

    run = lit.run.create_run(tests, lit_config, opts.workers, progress_callback,
                             opts.max_failures, opts.timeout)

    display.print_header()
    try:
        execute_in_tmp_dir(run, lit_config)
        display.clear(interrupted=False)
    except KeyboardInterrupt:
        display.clear(interrupted=True)
        print(' [interrupted by user]')


def execute_in_tmp_dir(run, lit_config):
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
        run.execute()
    finally:
        if tmp_dir:
            try:
                import shutil
                shutil.rmtree(tmp_dir)
            except:
                # FIXME: Re-try after timeout on Windows.
                lit_config.warning("Failed to delete temp directory '%s'" % tmp_dir)

def print_summary(tests, elapsed, opts):
    if not opts.quiet:
        print('\nTesting Time: %.2fs' % elapsed)

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
           (lit.Test.UNRESOLVED == code and (opts.max_failures is not None)):
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

def write_test_results(tests, lit_config, elapsed, output_path):
    # TODO(yln): audit: unexecuted tests
    # Construct the data we will write.
    data = {}
    # Encode the current lit version as a schema version.
    data['__version__'] = lit.__versioninfo__
    data['elapsed'] = elapsed
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
    # TODO(yln): audit: unexecuted tests
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
        if result_test.isFailure():
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
