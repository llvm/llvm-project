import inspect
import optparse
import shlex
import sys
import time

import lldb


class ProgressTesterCommand:
    program = "test-progress"

    @classmethod
    def register_lldb_command(cls, debugger, module_name):
        parser = cls.create_options()
        cls.__doc__ = parser.format_help()
        # Add any commands contained in this module to LLDB
        command = "command script add -c %s.%s %s" % (
            module_name,
            cls.__name__,
            cls.program,
        )
        debugger.HandleCommand(command)
        print(
            'The "{0}" command has been installed, type "help {0}" or "{0} '
            '--help" for detailed help.'.format(cls.program)
        )

    @classmethod
    def create_options(cls):
        usage = "usage: %prog [options]"
        description = "SBProgress testing tool"
        # Opt parse is deprecated, but leaving this the way it is because it allows help formating
        # Additionally all our commands use optparse right now, ideally we migrate them all in one go.
        parser = optparse.OptionParser(
            description=description, prog=cls.program, usage=usage
        )

        parser.add_option(
            "--total",
            dest="total",
            help="Total items in this progress object. When this option is not specified, this will be an indeterminate progress.",
            type="int",
            default=None,
        )

        parser.add_option(
            "--seconds",
            dest="seconds",
            help="Total number of seconds to wait between increments",
            type="int",
        )

        parser.add_option(
            "--no-details",
            dest="no_details",
            help="Do not display details",
            action="store_true",
            default=False,
        )

        return parser

    def get_short_help(self):
        return "Progress Tester"

    def get_long_help(self):
        return self.help_string

    def __init__(self, debugger, unused):
        self.parser = self.create_options()
        self.help_string = self.parser.format_help()

    def __call__(self, debugger, command, exe_ctx, result):
        command_args = shlex.split(command)
        try:
            (cmd_options, args) = self.parser.parse_args(command_args)
        except:
            result.SetError("option parsing failed")
            return

        total = cmd_options.total
        if total is None:
            progress = lldb.SBProgress(
                "Progress tester", "Initial Indeterminate Detail", debugger
            )
        else:
            progress = lldb.SBProgress(
                "Progress tester", "Initial Detail", total, debugger
            )

        # Check to see if total is set to None to indicate an indeterminate progress
        # then default to 10 steps.
        if total is None:
            total = 10

        for i in range(1, total):
            if cmd_options.no_details:
                progress.Increment(1)
            else:
                progress.Increment(1, f"Step {i}")
            time.sleep(cmd_options.seconds)

        # Not required for deterministic progress, but required for indeterminate progress.
        progress.Finalize()


def __lldb_init_module(debugger, dict):
    # Register all classes that have a register_lldb_command method
    for _name, cls in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(cls) and callable(
            getattr(cls, "register_lldb_command", None)
        ):
            cls.register_lldb_command(debugger, __name__)
