import argparse
import os
import shlex
import sys

import lit.reports
import lit.util


def parse_args():
    parser = argparse.ArgumentParser(prog='lit')
    parser.add_argument('test_paths',
            nargs='+',
            metavar="TEST_PATH",
            help='File or path to include in the test suite')

    parser.add_argument('--version',
            action='version',
            version='%(prog)s ' + lit.__version__)

    parser.add_argument("-j", "--threads", "--workers",
            dest="workers",
            metavar="N",
            help="Number of workers used for testing",
            type=_positive_int,
            default=lit.util.detectCPUs())
    parser.add_argument("--config-prefix",
            dest="configPrefix",
            metavar="NAME",
            help="Prefix for 'lit' config files")
    parser.add_argument("-D", "--param",
            dest="user_params",
            metavar="NAME=VAL",
            help="Add 'NAME' = 'VAL' to the user defined parameters",
            action="append",
            default=[])

    format_group = parser.add_argument_group("Output Format")
    # FIXME: I find these names very confusing, although I like the
    # functionality.
    format_group.add_argument("-q", "--quiet",
            help="Suppress no error output",
            action="store_true")
    format_group.add_argument("-s", "--succinct",
            help="Reduce amount of output",
            action="store_true")
    format_group.add_argument("-v", "--verbose",
            dest="showOutput",
            help="Show test output for failures",
            action="store_true")
    format_group.add_argument("-vv", "--echo-all-commands",
            dest="echoAllCommands",
            action="store_true",
            help="Echo all commands as they are executed to stdout. In case of "
                 "failure, last command shown will be the failing one.")
    format_group.add_argument("-a", "--show-all",
            dest="showAllOutput",
            help="Display all commandlines and output",
            action="store_true")
    format_group.add_argument("-o", "--output",
            type=lit.reports.JsonReport,
            help="Write test results to the provided path",
            metavar="PATH")
    format_group.add_argument("--no-progress-bar",
            dest="useProgressBar",
            help="Do not use curses based progress bar",
            action="store_false")
    format_group.add_argument("--show-unsupported",
            help="Show unsupported tests",
            action="store_true")
    format_group.add_argument("--show-xfail",
            help="Show tests that were expected to fail",
            action="store_true")

    execution_group = parser.add_argument_group("Test Execution")
    execution_group.add_argument("--path",
            help="Additional paths to add to testing environment",
            action="append",
            default=[])
    execution_group.add_argument("--vg",
            dest="useValgrind",
            help="Run tests under valgrind",
            action="store_true")
    execution_group.add_argument("--vg-leak",
            dest="valgrindLeakCheck",
            help="Check for memory leaks under valgrind",
            action="store_true")
    execution_group.add_argument("--vg-arg",
            dest="valgrindArgs",
            metavar="ARG",
            help="Specify an extra argument for valgrind",
            action="append",
            default=[])
    execution_group.add_argument("--time-tests",
            help="Track elapsed wall time for each test",
            action="store_true")
    execution_group.add_argument("--no-execute",
            dest="noExecute",
            help="Don't execute any tests (assume PASS)",
            action="store_true")
    execution_group.add_argument("--xunit-xml-output",
            type=lit.reports.XunitReport,
            help="Write XUnit-compatible XML test reports to the specified file")
    execution_group.add_argument("--timeout",
            dest="maxIndividualTestTime",
            help="Maximum time to spend running a single test (in seconds). "
                 "0 means no time limit. [Default: 0]",
            type=_non_negative_int) # TODO(yln): --[no-]test-timeout, instead of 0 allowed
    execution_group.add_argument("--max-failures",
            help="Stop execution after the given number of failures.",
            type=_positive_int)
    execution_group.add_argument("--allow-empty-runs",
            help="Do not fail the run if all tests are filtered out",
            action="store_true")

    selection_group = parser.add_argument_group("Test Selection")
    selection_group.add_argument("--max-tests",
            metavar="N",
            help="Maximum number of tests to run",
            type=_positive_int)
    selection_group.add_argument("--max-time", #TODO(yln): --timeout
            dest="timeout",
            metavar="N",
            help="Maximum time to spend testing (in seconds)",
            type=_positive_int)
    selection_group.add_argument("--shuffle",   # TODO(yln): --order=random
            help="Run tests in random order",   # default or 'by-path' (+ isEarlyTest())
            action="store_true")
    selection_group.add_argument("-i", "--incremental",  # TODO(yln): --order=failing-first
            help="Run modified and failing tests first (updates mtimes)",
            action="store_true")
    selection_group.add_argument("--filter",
            metavar="REGEX",
            type=_case_insensitive_regex,
            help="Only run tests with paths matching the given regular expression",
            default=os.environ.get("LIT_FILTER", ".*"))
    selection_group.add_argument("--num-shards", # TODO(yln): --shards N/M
            dest="numShards",
            metavar="M",
            help="Split testsuite into M pieces and only run one",
            type=_positive_int,
            default=os.environ.get("LIT_NUM_SHARDS"))
    selection_group.add_argument("--run-shard",
            dest="runShard",
            metavar="N",
            help="Run shard #N of the testsuite",
            type=_positive_int,
            default=os.environ.get("LIT_RUN_SHARD"))

    debug_group = parser.add_argument_group("Debug and Experimental Options")
    debug_group.add_argument("--debug",
            help="Enable debugging (for 'lit' development)",
            action="store_true")
    debug_group.add_argument("--show-suites",
            help="Show discovered test suites and exit",
            action="store_true")
    debug_group.add_argument("--show-tests",
            help="Show all discovered tests and exit",
            action="store_true")
    debug_group.add_argument("--show-used-features",
            help="Show all features used in the test suite (in XFAIL, UNSUPPORTED and REQUIRES) and exit",
            action="store_true")

    # LIT is special: environment variables override command line arguments.
    env_args = shlex.split(os.environ.get("LIT_OPTS", ""))
    args = sys.argv[1:] + env_args
    opts = parser.parse_args(args)

    # Validate command line options
    if opts.echoAllCommands:
        opts.showOutput = True

    # TODO(python3): Could be enum
    if opts.shuffle:
        opts.order = 'random'
    elif opts.incremental:
        opts.order = 'failing-first'
    else:
        opts.order = 'default'

    if opts.numShards or opts.runShard:
        if not opts.numShards or not opts.runShard:
            parser.error("--num-shards and --run-shard must be used together")
        if opts.runShard > opts.numShards:
            parser.error("--run-shard must be between 1 and --num-shards (inclusive)")
        opts.shard = (opts.runShard, opts.numShards)
    else:
        opts.shard = None

    opts.reports = filter(None, [opts.output, opts.xunit_xml_output])

    return opts


def _positive_int(arg):
    return _int(arg, 'positive', lambda i: i > 0)


def _non_negative_int(arg):
    return _int(arg, 'non-negative', lambda i: i >= 0)


def _int(arg, kind, pred):
    desc = "requires {} integer, but found '{}'"
    try:
        i = int(arg)
    except ValueError:
        raise _error(desc, kind, arg)
    if not pred(i):
        raise _error(desc, kind, arg)
    return i


def _case_insensitive_regex(arg):
    import re
    try:
        return re.compile(arg, re.IGNORECASE)
    except re.error as reason:
        raise _error("invalid regular expression: '{}', {}", arg, reason)


def _error(desc, *args):
    msg = desc.format(*args)
    return argparse.ArgumentTypeError(msg)
