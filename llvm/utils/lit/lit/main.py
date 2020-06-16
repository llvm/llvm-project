"""
lit - LLVM Integrated Tester.

See lit.pod for more information.
"""

import itertools
import os
import platform
import sys
import time

import lit.cl_arguments
import lit.discovery
import lit.display
import lit.LitConfig
import lit.reports
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

    if opts.show_used_features:
        features = set(itertools.chain.from_iterable(t.getUsedFeatures() for t in discovered_tests))
        print(' '.join(sorted(features)))
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

    determine_order(discovered_tests, opts.order)

    selected_tests = [t for t in discovered_tests if
                      opts.filter.search(t.getFullName())]
    if not selected_tests:
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

    if opts.shard:
        (run, shards) = opts.shard
        selected_tests = filter_by_shard(selected_tests, run, shards, lit_config)
        if not selected_tests:
            sys.stderr.write('warning: shard does not contain any tests.  '
                             'Consider decreasing the number of shards.\n')
            sys.exit(0)

    selected_tests = selected_tests[:opts.max_tests]

    mark_excluded(discovered_tests, selected_tests)

    start = time.time()
    run_tests(selected_tests, lit_config, opts, len(discovered_tests))
    elapsed = time.time() - start

    if opts.time_tests:
        print_histogram(discovered_tests)

    print_results(discovered_tests, elapsed, opts)

    for report in opts.reports:
        report.write_results(discovered_tests, elapsed)

    if lit_config.numErrors:
        sys.stderr.write('\n%d error(s) in tests\n' % lit_config.numErrors)
        sys.exit(2)

    if lit_config.numWarnings:
        sys.stderr.write('\n%d warning(s) in tests\n' % lit_config.numWarnings)

    has_failure = any(t.isFailure() for t in discovered_tests)
    if has_failure:
        sys.exit(1)


def create_params(builtin_params, user_params):
    def parse(p):
        return p.split('=', 1) if '=' in p else (p, '')

    params = dict(builtin_params)
    params.update([parse(p) for p in user_params])
    return params


def print_discovered(tests, show_suites, show_tests):
    tests.sort(key=lit.reports.by_suite_and_test_path)

    if show_suites:
        tests_by_suite = itertools.groupby(tests, lambda t: t.suite)
        print('-- Test Suites --')
        for suite, test_iter in tests_by_suite:
            test_count = sum(1 for _ in test_iter)
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


def mark_excluded(discovered_tests, selected_tests):
    excluded_tests = set(discovered_tests) - set(selected_tests)
    result = lit.Test.Result(lit.Test.EXCLUDED)
    for t in excluded_tests:
        t.setResult(result)


def run_tests(tests, lit_config, opts, discovered_tests):
    workers = min(len(tests), opts.workers)
    display = lit.display.create_display(opts, len(tests), discovered_tests,
                                         workers)

    def progress_callback(test):
        display.update(test)
        if opts.order == 'failing-first':
            touch_file(test)

    run = lit.run.Run(tests, lit_config, workers, progress_callback,
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
            except Exception as e: 
                lit_config.warning("Failed to delete temp directory '%s', try upgrading your version of Python to fix this" % tmp_dir)


def print_histogram(tests):
    test_times = [(t.getFullName(), t.result.elapsed)
                  for t in tests if t.result.elapsed]
    if test_times:
        lit.util.printHistogram(test_times, title='Tests')


def add_result_category(result_code, label):
    assert isinstance(result_code, lit.Test.ResultCode)
    category = (result_code, label)
    result_codes.append(category)


result_codes = [
    # Passes
    (lit.Test.EXCLUDED,    'Excluded'),
    (lit.Test.SKIPPED,     'Skipped'),
    (lit.Test.UNSUPPORTED, 'Unsupported'),
    (lit.Test.PASS,        'Passed'),
    (lit.Test.FLAKYPASS,   'Passed With Retry'),
    (lit.Test.XFAIL,       'Expectedly Failed'),
    # Failures
    (lit.Test.UNRESOLVED,  'Unresolved'),
    (lit.Test.TIMEOUT,     'Timed Out'),
    (lit.Test.FAIL,        'Failed'),
    (lit.Test.XPASS,       'Unexpectedly Passed')
]


def print_results(tests, elapsed, opts):
    tests_by_code = {code: [] for code, _ in result_codes}
    for test in tests:
        tests_by_code[test.result.code].append(test)

    for (code, label) in result_codes:
        print_group(code, label, tests_by_code[code], opts.show_results)

    print_summary(tests_by_code, opts.quiet, elapsed)


def print_group(code, label, tests, show_results):
    if not tests:
        return
    if not code.isFailure and code not in show_results:
        return
    print('*' * 20)
    print('%s Tests (%d):' % (label, len(tests)))
    for test in tests:
        print('  %s' % test.getFullName())
    sys.stdout.write('\n')


def print_summary(tests_by_code, quiet, elapsed):
    if not quiet:
        print('\nTesting Time: %.2fs' % elapsed)

    codes = [c for c in result_codes if not quiet or c.isFailure]
    groups = [(label, len(tests_by_code[code])) for code, label in codes]
    groups = [(label, count) for label, count in groups if count]
    if not groups:
        return

    max_label_len = max(len(label) for label, _ in groups)
    max_count_len = max(len(str(count)) for _, count in groups)

    for (label, count) in groups:
        label = label.ljust(max_label_len)
        count = str(count).rjust(max_count_len)
        print('  %s: %s' % (label, count))
