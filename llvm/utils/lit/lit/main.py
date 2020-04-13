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

    if opts.show_suites or opts.show_tests:
        print_discovered(discovered_tests, opts.show_suites, opts.show_tests)
        sys.exit(0)

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

    filtered_tests = [t for t in discovered_tests if
                      opts.filter.search(t.getFullName())]
    if not filtered_tests:
        sys.stderr.write('error: filter did not match any tests '
                         '(of %d discovered).  ' % len(discovered_tests))
        if opts.allow_empty_runs:
            sys.stderr.write("Suppressing error because '--allow-empty-runs' "
                             'was specified.\n')
            sys.exit(0)
        else:
            sys.stderr.write("Use '--allow-empty-runs' to suppress this "
                             'error.\n')
            sys.exit(2)

    determine_order(filtered_tests, opts.order)

    if opts.shard:
        (run, shards) = opts.shard
        filtered_tests = filter_by_shard(filtered_tests, run, shards, lit_config)
        if not filtered_tests:
            sys.stderr.write('warning: shard does not contain any tests.  '
                             'Consider decreasing the number of shards.\n')
            sys.exit(0)

    filtered_tests = filtered_tests[:opts.max_tests]

    opts.workers = min(len(filtered_tests), opts.workers)

    start = time.time()
    run_tests(filtered_tests, lit_config, opts, len(discovered_tests))
    elapsed = time.time() - start

    # TODO(yln): eventually, all functions below should act on discovered_tests
    executed_tests = [
        t for t in filtered_tests if t.result.code != lit.Test.SKIPPED]

    if opts.time_tests:
        print_histogram(executed_tests)

    print_results(filtered_tests, elapsed, opts)

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


def print_discovered(tests, show_suites, show_tests):
    # Suite names are not necessarily unique.  Include object identity in sort
    # key to avoid mixing tests of different suites.
    tests.sort(key=lambda t: (t.suite.name, t.suite, t.path_in_suite))

    if show_suites:
        import itertools
        tests_by_suite = itertools.groupby(tests, lambda t: t.suite)
        print('-- Test Suites --')
        for suite, suite_iter in tests_by_suite:
            test_count = sum(1 for _ in suite_iter)
            print('  %s - %d tests' % (suite.name, test_count))
            print('    Source Root: %s' % suite.source_root)
            print('    Exec Root  : %s' % suite.exec_root)
            features = ' '.join(sorted(suite.config.available_features))
            print('    Available Features: %s' % features)
            substitutions = sorted(suite.config.substitutions)
            substitutions = ('%s => %s' % (x, y) for (x, y) in substitutions)
            substitutions = '\n'.ljust(30).join(substitutions)
            print('    Available Substitutions: %s' % substitutions)

    if show_tests:
        print('-- Available Tests --')
        for t in tests:
            print('  %s' % t.getFullName())


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


def run_tests(tests, lit_config, opts, discovered_tests):
    display = lit.display.create_display(opts, len(tests), discovered_tests,
                                         opts.workers)
    def progress_callback(test):
        display.update(test)
        if opts.order == 'failing-first':
            touch_file(test)

    run = lit.run.Run(tests, lit_config, opts.workers, progress_callback,
                      opts.max_failures, opts.timeout)

    display.print_header()

    interrupted = False
    error = None
    try:
        execute_in_tmp_dir(run, lit_config)
    except KeyboardInterrupt:
        interrupted = True
        error = '  interrupted by user'
    except lit.run.MaxFailuresError:
        error = 'warning: reached maximum number of test failures'
    except lit.run.TimeoutError:
        error = 'warning: reached timeout'

    display.clear(interrupted)
    if error:
        sys.stderr.write('%s, skipping remaining tests\n' % error)


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


def print_histogram(tests):
    test_times = [(t.getFullName(), t.result.elapsed) for t in tests]
    lit.util.printHistogram(test_times, title='Tests')


# Status code, summary label, group label
failure_codes = [
    (lit.Test.UNRESOLVED,  'Unresolved Tests',    'Unresolved'),
    (lit.Test.TIMEOUT,     'Individual Timeouts', 'Timed Out'),
    (lit.Test.FAIL,        'Unexpected Failures', 'Failing'),
    (lit.Test.XPASS,       'Unexpected Passes',   'Unexpected Passing')
]

all_codes = [
    (lit.Test.SKIPPED,     'Skipped Tests',     'Skipped'),
    (lit.Test.UNSUPPORTED, 'Unsupported Tests', 'Unsupported'),
    (lit.Test.PASS,        'Expected Passes',   ''),
    (lit.Test.FLAKYPASS,   'Passes With Retry', ''),
    (lit.Test.XFAIL,       'Expected Failures', 'Expected Failing'),
] + failure_codes


def print_results(tests, elapsed, opts):
    tests_by_code = {code: [] for (code, _, _) in all_codes}
    for test in tests:
        tests_by_code[test.result.code].append(test)

    for (code, _, group_label) in all_codes:
        print_group(code, group_label, tests_by_code[code], opts)

    print_summary(tests_by_code, opts.quiet, elapsed)


def print_group(code, label, tests, opts):
    if not tests:
        return
    # TODO(yln): FLAKYPASS? Make this more consistent!
    if code in {lit.Test.SKIPPED, lit.Test.PASS}:
        return
    if (lit.Test.XFAIL == code and not opts.show_xfail) or \
       (lit.Test.UNSUPPORTED == code and not opts.show_unsupported):
        return
    print('*' * 20)
    print('%s Tests (%d):' % (label, len(tests)))
    for test in tests:
        print('  %s' % test.getFullName())
    sys.stdout.write('\n')


def print_summary(tests_by_code, quiet, elapsed):
    if not quiet:
        print('\nTesting Time: %.2fs' % elapsed)

    codes = failure_codes if quiet else all_codes
    groups = [(label, len(tests_by_code[code])) for code, label, _ in codes]
    groups = [(label, count) for label, count in groups if count]
    if not groups:
        return

    max_label_len = max(len(label) for label, _ in groups)
    max_count_len = max(len(str(count)) for _, count in groups)

    for (label, count) in groups:
        label = label.ljust(max_label_len)
        count = str(count).rjust(max_count_len)
        print('  %s: %s' % (label, count))


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
