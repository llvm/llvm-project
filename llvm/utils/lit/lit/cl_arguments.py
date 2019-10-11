import argparse
import os
import shlex
import sys

import lit.util

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_paths',
            nargs='+',
            help='Files or paths to include in the test suite')

    parser.add_argument("--version",
            dest="show_version",
            help="Show version and exit",
            action="store_true",
            default=False)
    parser.add_argument("-j", "--threads", "--workers",
            dest="numWorkers",
            metavar="N",
            help="Number of workers used for testing",
            type=_positive_int,
            default=lit.util.detectCPUs())
    parser.add_argument("--config-prefix",
            dest="configPrefix",
            metavar="NAME",
            help="Prefix for 'lit' config files",
            default=None)
    parser.add_argument("-D", "--param",
            dest="userParameters",
            metavar="NAME=VAL",
            help="Add 'NAME' = 'VAL' to the user defined parameters",
            type=str,
            action="append",
            default=[])

    format_group = parser.add_argument_group("Output Format")
    # FIXME: I find these names very confusing, although I like the
    # functionality.
    format_group.add_argument("-q", "--quiet",
            help="Suppress no error output",
            action="store_true",
            default=False)
    format_group.add_argument("-s", "--succinct",
            help="Reduce amount of output",
            action="store_true",
            default=False)
    format_group.add_argument("-v", "--verbose",
            dest="showOutput",
            help="Show test output for failures",
            action="store_true",
            default=False)
    format_group.add_argument("-vv", "--echo-all-commands",
            dest="echoAllCommands",
            action="store_true",
            default=False,
            help="Echo all commands as they are executed to stdout. In case of "
                 "failure, last command shown will be the failing one.")
    format_group.add_argument("-a", "--show-all",
            dest="showAllOutput",
            help="Display all commandlines and output",
            action="store_true",
            default=False)
    format_group.add_argument("-o", "--output",
            dest="output_path",
            help="Write test results to the provided path",
            metavar="PATH")
    format_group.add_argument("--no-progress-bar",
            dest="useProgressBar",
            help="Do not use curses based progress bar",
            action="store_false",
            default=True)
    format_group.add_argument("--show-unsupported",
            help="Show unsupported tests",
            action="store_true",
            default=False)
    format_group.add_argument("--show-xfail",
            help="Show tests that were expected to fail",
            action="store_true",
            default=False)

    execution_group = parser.add_argument_group("Test Execution")
    execution_group.add_argument("--path",
            help="Additional paths to add to testing environment",
            action="append",
            type=str,
            default=[])
    execution_group.add_argument("--vg",
            dest="useValgrind",
            help="Run tests under valgrind",
            action="store_true",
            default=False)
    execution_group.add_argument("--vg-leak",
            dest="valgrindLeakCheck",
            help="Check for memory leaks under valgrind",
            action="store_true",
            default=False)
    execution_group.add_argument("--vg-arg",
            dest="valgrindArgs",
            metavar="ARG",
            help="Specify an extra argument for valgrind",
            type=str,
            action="append",
            default=[])
    execution_group.add_argument("--time-tests",
            dest="timeTests",
            help="Track elapsed wall time for each test",
            action="store_true",
            default=False)
    execution_group.add_argument("--no-execute",
            dest="noExecute",
            help="Don't execute any tests (assume PASS)",
            action="store_true",
            default=False)
    execution_group.add_argument("--xunit-xml-output",
            dest="xunit_output_file",
            help="Write XUnit-compatible XML test reports to the specified file",
            default=None)
    execution_group.add_argument("--timeout",
            dest="maxIndividualTestTime",
            help="Maximum time to spend running a single test (in seconds). "
                 "0 means no time limit. [Default: 0]",
            type=int,
            default=None)
    execution_group.add_argument("--max-failures",
            dest="maxFailures",
            help="Stop execution after the given number of failures.",
            type=_positive_int,
            default=None)

    selection_group = parser.add_argument_group("Test Selection")
    selection_group.add_argument("--max-tests",
            dest="maxTests",
            metavar="N",
            help="Maximum number of tests to run",
            type=int,
            default=None)
    selection_group.add_argument("--max-time",
            dest="maxTime",
            metavar="N",
            help="Maximum time to spend testing (in seconds)",
            type=float,
            default=None)
    selection_group.add_argument("--shuffle",
            help="Run tests in random order",
            action="store_true",
            default=False)
    selection_group.add_argument("-i", "--incremental",
            help="Run modified and failing tests first (updates mtimes)",
            action="store_true",
            default=False)
    selection_group.add_argument("--filter",
            metavar="REGEX",
            help="Only run tests with paths matching the given regular expression",
            default=os.environ.get("LIT_FILTER"))
    selection_group.add_argument("--num-shards",
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
            action="store_true",
            default=False)
    debug_group.add_argument("--show-suites",
            dest="showSuites",
            help="Show discovered test suites",
            action="store_true",
            default=False)
    debug_group.add_argument("--show-tests",
            dest="showTests",
            help="Show all discovered tests",
            action="store_true",
            default=False)

    # LIT is special: environment variables override command line arguments.
    env_args = shlex.split(os.environ.get("LIT_OPTS", ""))
    args = sys.argv[1:] + env_args
    opts = parser.parse_args(args)

    # Validate command line options
    if opts.echoAllCommands:
        opts.showOutput = True

    if opts.numShards or opts.runShard:
        if not opts.numShards or not opts.runShard:
            parser.error("--num-shards and --run-shard must be used together")
        if opts.runShard > opts.numShards:
            parser.error("--run-shard must be between 1 and --num-shards (inclusive)")

    return opts

def _positive_int(arg):
    try:
        n = int(arg)
    except ValueError:
        raise _arg_error('positive integer', arg)
    if n <= 0:
        raise _arg_error('positive integer', arg)
    return n

def _arg_error(desc, arg):
    msg = "requires %s, but found '%s'" % (desc, arg)
    return argparse.ArgumentTypeError(msg)
