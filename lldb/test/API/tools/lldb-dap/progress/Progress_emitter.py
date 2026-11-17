import optparse
import shlex
import time

import lldb


def make_parser():
    parser = optparse.OptionParser(
        prog="send-progress",
        description="SBProgress testing tool",
        usage="usage: %prog [options]",
    )
    parser.add_option(
        "--total",
        type="int",
        default=None,
        help="Total items in this progress object. Omit for indeterminate progress.",
    )
    parser.add_option(
        "--seconds",
        type="float",
        default=0.0,
        help="Seconds to sleep between increments.",
    )
    parser.add_option(
        "--no-details",
        action="store_true",
        default=False,
        help="Do not attach a per-step detail string.",
    )
    return parser


class SendProgressCommand:
    """Drive an lldb.SBProgress for lldb-dap tests."""

    def __init__(self, debugger, internal_dict):
        pass

    def __call__(self, debugger, command, exe_ctx, result):
        try:
            parser = make_parser()
            opts, _ = parser.parse_args(shlex.split(command))
        except SystemExit:
            result.SetError("option parsing failed")
            return

        if opts.total is None:
            progress = lldb.SBProgress(
                "Progress tester", "Initial Indeterminate Detail", debugger
            )
            iterations = 5
        else:
            progress = lldb.SBProgress(
                "Progress tester", "Initial Detail", opts.total, debugger
            )
            iterations = opts.total - 1

        with progress:
            for i in range(iterations):
                if opts.no_details:
                    progress.Increment(1)
                else:
                    progress.Increment(1, f"Step {i}")
                time.sleep(opts.seconds)


def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand(
        f"command script add -c {__name__}.SendProgressCommand send-progress"
    )
