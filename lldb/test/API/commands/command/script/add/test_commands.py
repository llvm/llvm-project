"""
Test defining commands using the lldb command definitions
"""
import inspect
import sys
import lldb
from lldb.plugins.parsed_cmd import ParsedCommand


class ReportingCmd(ParsedCommand):
    def __init__(self, debugger, unused):
        super().__init__(debugger, unused)

    def __call__(self, debugger, args_array, exe_ctx, result):
        opt_def = self.get_options_definition()
        if len(opt_def):
            result.AppendMessage("Options:\n")
            for long_option, elem in opt_def.items():
                dest = elem["dest"]
                result.AppendMessage(
                    f"{long_option} (set: {elem['_value_set']}): {object.__getattribute__(self.ov_parser, dest)}\n"
                )
        else:
            result.AppendMessage("No options\n")

        num_args = args_array.GetSize()
        if num_args > 0:
            result.AppendMessage(f"{num_args} arguments:")
        for idx in range(0, num_args):
            result.AppendMessage(
                f"{idx}: {args_array.GetItemAtIndex(idx).GetStringValue(10000)}\n"
            )


class NoArgsCommand(ReportingCmd):
    program = "no-args"

    def __init__(self, debugger, unused):
        super().__init__(debugger, unused)

    @classmethod
    def register_lldb_command(cls, debugger, module_name):
        ParsedCommand.do_register_cmd(cls, debugger, module_name)

    def setup_command_definition(self):
        self.ov_parser.add_option(
            "b",
            "bool-arg",
            "a boolean arg, defaults to True",
            value_type=lldb.eArgTypeBoolean,
            groups=[1, 2],
            dest="bool_arg",
            default=True,
        )

        self.ov_parser.add_option(
            "s",
            "shlib-name",
            "A shared library name.",
            value_type=lldb.eArgTypeShlibName,
            groups=[1, [3, 4]],
            dest="shlib_name",
            default=None,
        )

        self.ov_parser.add_option(
            "d",
            "disk-file-name",
            "An on disk filename",
            value_type=lldb.eArgTypeFilename,
            dest="disk_file_name",
            default=None,
        )

        self.ov_parser.add_option(
            "l",
            "line-num",
            "A line number",
            value_type=lldb.eArgTypeLineNum,
            groups=3,
            dest="line_num",
            default=0,
        )

        self.ov_parser.add_option(
            "e",
            "enum-option",
            "An enum, doesn't actually do anything",
            enum_values=[
                ["foo", "does foo things"],
                ["bar", "does bar things"],
                ["baz", "does baz things"],
            ],
            groups=4,
            dest="enum_option",
            default="foo",
        )

    def get_short_help(self):
        return "Example command for use in debugging"

    def get_long_help(self):
        return self.help_string


class OneArgCommandNoOptions(ReportingCmd):
    program = "one-arg-no-opt"

    def __init__(self, debugger, unused):
        super().__init__(debugger, unused)

    @classmethod
    def register_lldb_command(cls, debugger, module_name):
        ParsedCommand.do_register_cmd(cls, debugger, module_name)

    def setup_command_definition(self):
        self.ov_parser.add_argument_set(
            [self.ov_parser.make_argument_element(lldb.eArgTypeSourceFile, "plain")]
        )

    def get_short_help(self):
        return "Example command for use in debugging"

    def get_long_help(self):
        return self.help_string


class TwoArgGroupsCommand(ReportingCmd):
    program = "two-args"

    def __init__(self, debugger, unused):
        super().__init__(debugger, unused)

    @classmethod
    def register_lldb_command(cls, debugger, module_name):
        ParsedCommand.do_register_cmd(cls, debugger, module_name)

    def setup_command_definition(self):
        self.ov_parser.add_option(
            "l",
            "language",
            "language defaults to None",
            value_type=lldb.eArgTypeLanguage,
            groups=[1, 2],
            dest="language",
            default=None,
        )

        self.ov_parser.add_option(
            "c",
            "log-channel",
            "log channel - defaults to lldb",
            value_type=lldb.eArgTypeLogChannel,
            groups=[1, 3],
            dest="log_channel",
            default="lldb",
        )

        self.ov_parser.add_option(
            "p",
            "process-name",
            "A process name, defaults to None",
            value_type=lldb.eArgTypeProcessName,
            dest="proc_name",
            default=None,
        )

        self.ov_parser.add_argument_set(
            [
                self.ov_parser.make_argument_element(
                    lldb.eArgTypeClassName, "plain", [1, 2]
                ),
                self.ov_parser.make_argument_element(
                    lldb.eArgTypeOffset, "optional", [1, 2]
                ),
            ]
        )

        self.ov_parser.add_argument_set(
            [
                self.ov_parser.make_argument_element(
                    lldb.eArgTypePythonClass, "plain", [3, 4]
                ),
                self.ov_parser.make_argument_element(
                    lldb.eArgTypePid, "optional", [3, 4]
                ),
            ]
        )

    def get_short_help(self):
        return "Example command for use in debugging"

    def get_long_help(self):
        return self.help_string


def __lldb_init_module(debugger, dict):
    # Register all classes that have a register_lldb_command method
    for _name, cls in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(cls) and callable(
            getattr(cls, "register_lldb_command", None)
        ):
            cls.register_lldb_command(debugger, __name__)
