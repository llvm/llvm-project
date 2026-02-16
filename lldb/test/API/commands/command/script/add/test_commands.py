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
                if "value_type" in elem:
                    print(f"Looking at {long_option} - {elem}")
                    dest = elem["dest"]
                    result.AppendMessage(
                        f"{long_option} (set: {elem['_value_set']}): {object.__getattribute__(self.get_parser(), dest)}\n"
                    )
                else:
                    result.AppendMessage(
                        f"{long_option} (set: {elem['_value_set']}): flag value\n"
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

# Use these to make sure that get_repeat_command sends the right
# command.
no_args_repeat = None
one_arg_repeat = None
two_arg_repeat = None

class NoArgsCommand(ReportingCmd):
    program = "no-args"

    def __init__(self, debugger, unused):
        super().__init__(debugger, unused)

    @classmethod
    def register_lldb_command(cls, debugger, module_name):
        ParsedCommand.do_register_cmd(cls, debugger, module_name)

    def setup_command_definition(self):
        ov_parser = self.get_parser()
        ov_parser.add_option(
            "b",
            "bool-arg",
            "a boolean arg, defaults to True",
            value_type=lldb.eArgTypeBoolean,
            groups=[1, 2],
            dest="bool_arg",
            default=True,
        )

        ov_parser.add_option(
            "s",
            "shlib-name",
            "A shared library name.",
            value_type=lldb.eArgTypeShlibName,
            groups=[1, [3, 4]],
            dest="shlib_name",
            default=None,
        )

        ov_parser.add_option(
            "d",
            "disk-file-name",
            "An on disk filename",
            value_type=lldb.eArgTypeFilename,
            dest="disk_file_name",
            default=None,
        )

        ov_parser.add_option("f", "flag-value", "This is a flag value")

        ov_parser.add_option(
            "l",
            "line-num",
            "A line number",
            value_type=lldb.eArgTypeLineNum,
            groups=3,
            dest="line_num",
            default=0,
        )

        ov_parser.add_option(
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

    def get_repeat_command(self, command):
        # No auto-repeat
        global no_args_repeat
        no_args_repeat = command
        return ""

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
        ov_parser = self.get_parser()
        ov_parser.add_argument_set(
            [ov_parser.make_argument_element(lldb.eArgTypeSourceFile, "plain")]
        )

    def get_repeat_command(self, command):
        # Repeat the current command
        global one_arg_repeat
        one_arg_repeat = command
        return None

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
        ov_parser = self.get_parser()
        ov_parser.add_option(
            "l",
            "language",
            "language defaults to None",
            value_type=lldb.eArgTypeLanguage,
            groups=[1, 2],
            dest="language",
            default=None,
        )

        ov_parser.add_option(
            "c",
            "log-channel",
            "log channel - defaults to lldb",
            value_type=lldb.eArgTypeLogChannel,
            groups=[1, 3],
            dest="log_channel",
            default="lldb",
        )

        ov_parser.add_option(
            "p",
            "process-name",
            "A process name, defaults to None",
            value_type=lldb.eArgTypeProcessName,
            dest="proc_name",
            default=None,
        )

        ov_parser.add_argument_set(
            [
                ov_parser.make_argument_element(
                    lldb.eArgTypeClassName, "plain", [1, 2]
                ),
                ov_parser.make_argument_element(
                    lldb.eArgTypeOffset, "optional", [1, 2]
                ),
            ]
        )

        ov_parser.add_argument_set(
            [
                ov_parser.make_argument_element(
                    lldb.eArgTypePythonClass, "plain", [3, 4]
                ),
                ov_parser.make_argument_element(lldb.eArgTypePid, "optional", [3, 4]),
            ]
        )

    def get_repeat_command(self, command):
        global two_arg_repeat
        two_arg_repeat = command
        return command + " THIRD_ARG"

    def handle_option_argument_completion(self, long_option, cursor_pos):
        ov_parser = self.get_parser()
        value = ov_parser.dest_for_option(long_option)[0 : cursor_pos + 1]
        proc_value = ov_parser.proc_name
        if proc_value != None:
            new_str = value + proc_value
            ret_arr = {"completion": new_str, "mode": "partial"}
            return ret_arr

        ret_arr = {"values": [value + "nice", value + "not_nice", value + "mediocre"]}
        return ret_arr

    def handle_argument_completion(self, args, arg_pos, cursor_pos):
        ov_parser = self.get_parser()
        orig_arg = args[arg_pos][0:cursor_pos]
        if orig_arg == "correct_":
            ret_arr = {"completion": "correct_answer"}
            return ret_arr

        if ov_parser.was_set("process-name"):
            # No completions if proc_name was set.
            return True

        ret_arr = {
            "values": [orig_arg + "cool", orig_arg + "yuck"],
            "descriptions": ["good idea", "bad idea"],
        }
        return ret_arr

    def get_short_help(self):
        return "This is my short help string"

    def get_long_help(self):
        return self.help_string


def __lldb_init_module(debugger, dict):
    # Register all classes that have a register_lldb_command method
    for _name, cls in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(cls) and callable(
            getattr(cls, "register_lldb_command", None)
        ):
            cls.register_lldb_command(debugger, __name__)
