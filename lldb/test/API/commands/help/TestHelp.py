"""
Test some lldb help commands.

See also CommandInterpreter::OutputFormattedHelpText().
"""


import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class HelpCommandTestCase(TestBase):
    @no_debug_info_test
    def test_simplehelp(self):
        """A simple test of 'help' command and its output."""
        self.expect("help", startstr="Debugger commands:")

        self.expect("help -a", matching=False, substrs=["next"])

        self.expect("help", matching=True, substrs=["next"])

    @no_debug_info_test
    def test_help_on_help(self):
        """Testing the help on the help facility."""
        self.expect(
            "help help",
            matching=True,
            substrs=["--hide-aliases", "--hide-user-commands"],
        )

    @no_debug_info_test
    def test_help_arch(self):
        """Test 'help arch' which should list of supported architectures."""
        self.expect("help arch", substrs=["arm", "i386", "x86_64"])

    @no_debug_info_test
    def test_help_version(self):
        """Test 'help version' and 'version' commands."""
        self.expect("help version", substrs=["Show the LLDB debugger version."])
        self.expect("version", patterns=["lldb( version|-[0-9]+).*\n"])

    @no_debug_info_test
    def test_help_should_not_crash_lldb(self):
        """Command 'help disasm' should not crash lldb."""
        self.runCmd("help disasm", check=False)
        self.runCmd("help unsigned-integer")

    @no_debug_info_test
    def test_help_memory_read_should_not_crash_lldb(self):
        """Command 'help memory read' should not crash lldb."""
        self.runCmd("help memory read", check=False)

    @no_debug_info_test
    def test_help_should_not_hang_emacsshell(self):
        """Command 'settings set term-width 0' should not hang the help command."""
        self.expect(
            "settings set term-width 0",
            COMMAND_FAILED_AS_EXPECTED,
            error=True,
            substrs=["error: 0 is out of range, valid values must be between"],
        )
        # self.runCmd("settings set term-width 0")
        self.expect("help", startstr="Debugger commands:")

    @no_debug_info_test
    def test_help_breakpoint_set(self):
        """Test that 'help breakpoint set' does not print out redundant lines of:
        'breakpoint set [-s <shlib-name>] ...'."""
        self.expect(
            "help breakpoint set",
            matching=False,
            substrs=["breakpoint set [-s <shlib-name>]"],
        )

    @no_debug_info_test
    def test_help_image_dump_symtab_should_not_crash(self):
        """Command 'help image dump symtab' should not crash lldb."""
        # 'image' is an alias for 'target modules'.
        self.expect("help image dump symtab", substrs=["dump symtab", "sort-order"])

    @no_debug_info_test
    def test_help_image_du_sym_is_ambiguous(self):
        """Command 'help image du sym' is ambiguous and spits out the list of candidates."""
        self.expect(
            "help image du sym",
            COMMAND_FAILED_AS_EXPECTED,
            error=True,
            substrs=["error: ambiguous command image du sym", "symfile", "symtab"],
        )

    @no_debug_info_test
    def test_help_image_du_line_should_work(self):
        """Command 'help image du line-table' is not ambiguous and should work."""
        # 'image' is an alias for 'target modules'.
        self.expect(
            "help image du line",
            substrs=["Dump the line table for one or more compilation units"],
        )

    @no_debug_info_test
    def test_help_image_list_shows_positional_args(self):
        """Command 'help image list' should describe positional args."""
        # 'image' is an alias for 'target modules'.
        self.expect("help image list", substrs=["<shlib-name> [...]"])

    @no_debug_info_test
    def test_help_target_variable_syntax(self):
        """Command 'help target variable' should display <variable-name> ..."""
        self.expect(
            "help target variable", substrs=["<variable-name> [<variable-name> [...]]"]
        )

    @no_debug_info_test
    def test_help_watchpoint_and_its_args(self):
        """Command 'help watchpoint', 'help watchpt-id', and 'help watchpt-id-list' should work."""
        self.expect("help watchpoint", substrs=["delete", "disable", "enable", "list"])
        self.expect("help watchpt-id", substrs=["<watchpt-id>"])
        self.expect("help watchpt-id-list", substrs=["<watchpt-id-list>"])

    @no_debug_info_test
    def test_help_watchpoint_set(self):
        """Test that 'help watchpoint set' prints out 'expression' and 'variable'
        as the possible subcommands."""
        self.expect(
            "help watchpoint set",
            substrs=["The following subcommands are supported:"],
            patterns=["expression +--", "variable +--"],
        )

    @no_debug_info_test
    def test_help_po_hides_options(self):
        """Test that 'help po' does not show all the options for expression"""
        self.expect(
            "help po",
            substrs=["--show-all-children", "--object-description"],
            matching=False,
        )

    @no_debug_info_test
    def test_help_run_hides_options(self):
        """Test that 'help run' does not show all the options for process launch"""
        self.expect("help run", substrs=["--arch", "--environment"], matching=False)

    @no_debug_info_test
    def test_help_next_shows_options(self):
        """Test that 'help next' shows all the options for thread step-over"""
        self.expect(
            "help next",
            substrs=["--step-out-avoids-no-debug", "--run-mode"],
            matching=True,
        )

    @no_debug_info_test
    def test_help_provides_alternatives(self):
        """Test that help on commands that don't exist provides information on additional help avenues"""
        self.expect(
            "help thisisnotadebuggercommand",
            substrs=[
                "'thisisnotadebuggercommand' is not a known command.",
                "Try 'help' to see a current list of commands.",
                "Try 'apropos thisisnotadebuggercommand' for a list of related commands.",
                "Try 'type lookup thisisnotadebuggercommand' for information on types, methods, functions, modules, etc.",
            ],
            error=True,
        )

        self.expect(
            "help process thisisnotadebuggercommand",
            substrs=[
                "'process thisisnotadebuggercommand' is not a known command.",
                "Try 'help' to see a current list of commands.",
                "Try 'apropos thisisnotadebuggercommand' for a list of related commands.",
                "Try 'type lookup thisisnotadebuggercommand' for information on types, methods, functions, modules, etc.",
            ],
        )

    @no_debug_info_test
    def test_custom_help_alias(self):
        """Test that aliases pick up custom help text."""

        def cleanup():
            self.runCmd("command unalias afriendlyalias", check=False)
            self.runCmd("command unalias averyfriendlyalias", check=False)

        self.addTearDownHook(cleanup)
        self.runCmd(
            'command alias --help "I am a friendly alias" -- afriendlyalias help'
        )
        self.expect(
            "help afriendlyalias", matching=True, substrs=["I am a friendly alias"]
        )
        self.runCmd(
            'command alias --long-help "I am a very friendly alias" -- averyfriendlyalias help'
        )
        self.expect(
            "help averyfriendlyalias",
            matching=True,
            substrs=["I am a very friendly alias"],
        )

    @no_debug_info_test
    def test_alias_prints_origin(self):
        """Test that 'help <unique_match_to_alias>' prints the alias origin."""

        def cleanup():
            self.runCmd("command unalias alongaliasname", check=False)

        self.addTearDownHook(cleanup)
        self.runCmd("command alias alongaliasname help")
        self.expect(
            "help alongaliasna",
            matching=True,
            substrs=["'alongaliasna' is an abbreviation for 'help'"],
        )

    @no_debug_info_test
    def test_hidden_help(self):
        self.expect("help -h", substrs=["_regexp-bt"])

    @no_debug_info_test
    def test_help_ambiguous(self):
        self.expect(
            "help g",
            substrs=[
                "Help requested with ambiguous command name, possible completions:",
                "gdb-remote",
                "gui",
            ],
        )

    @no_debug_info_test
    def test_help_unknown_flag(self):
        self.expect("help -z", error=True, substrs=["unknown or ambiguous option"])

    @no_debug_info_test
    def test_help_format_output(self):
        """Test that help output reaches TerminalWidth."""
        self.runCmd("settings set term-width 108")
        self.expect(
            "help format",
            matching=True,
            substrs=["<format> -- One of the format names"],
        )

    @no_debug_info_test
    def test_help_option_group_format_options_usage(self):
        """Test that help on commands that use OptionGroupFormat options provide relevant help specific to that command."""
        self.expect(
            "help memory read",
            matching=True,
            substrs=[
                "-f <format> ( --format <format> )",
                "Specify a format to be used for display.",
                "-s <byte-size> ( --size <byte-size> )",
                "The size in bytes to use when displaying with the selected format.",
            ],
        )

        self.expect(
            "help memory write",
            matching=True,
            substrs=[
                "-f <format> ( --format <format> )",
                "The format to use for each of the value to be written.",
                "-s <byte-size> ( --size <byte-size> )",
                "The size in bytes to write from input file or each value.",
            ],
        )

    @no_debug_info_test
    def test_help_shows_optional_short_options(self):
        """Test that optional short options are printed and that they are in
        alphabetical order with upper case options first."""
        self.expect(
            "help memory read", substrs=["memory read [-br]", "memory read [-AFLORTr]"]
        )
        self.expect(
            "help target modules lookup", substrs=["target modules lookup [-Airv]"]
        )

    @no_debug_info_test
    def test_help_shows_command_options_usage(self):
        """Test that we start the usage section with a specific line."""
        self.expect(
            "help memory read", substrs=["Command Options Usage:\n  memory read"]
        )

    @no_debug_info_test
    def test_help_detailed_information_spacing(self):
        """Test that we put a break between the usage and the options help lines,
        and between the options themselves."""
        self.expect(
            "help memory read",
            substrs=[
                "[<address-expression>]\n\n       -A ( --show-all-children )",
                # Starts with the end of the show-all-children line
                "to show.\n\n       -D",
            ],
        )

    @no_debug_info_test
    def test_help_detailed_information_ordering(self):
        """Test that we order options alphabetically, upper case first."""
        # You could test this with a simple regex like:
        # <upper case>.*<later upper case>.*<lower case>.*<later lower case>
        # Except that that could pass sometimes even with shuffled output.
        # This makes sure that doesn't happen.

        self.runCmd("help memory read")
        got = self.res.GetOutput()
        _, options_lines = got.split("Command Options Usage:")
        options_lines = options_lines.lstrip().splitlines()

        # Skip over "memory read [-xyz] lines.
        while "memory read" in options_lines[0]:
            options_lines.pop(0)
        # Plus the newline after that.
        options_lines.pop(0)

        short_options = []
        for line in options_lines:
            # Ignore line breaks and descriptions.
            # (not stripping the line here in case some line of the descriptions
            # happens to start with "-")
            if not line or not line.startswith("       -"):
                continue
            # This apears at the end after the options.
            if "This command takes options and free form arguments." in line:
                break
            line = line.strip()
            # The order of -- only options is not enforced so ignore their position.
            if not line.startswith("--"):
                # Save its short char name.
                short_options.append(line[1])

        self.assertEqual(
            sorted(short_options),
            short_options,
            "Short option help displayed in an incorrect order!",
        )

    @no_debug_info_test
    def test_help_show_tags(self):
        """Check that memory find and memory read have the --show-tags option
        but only memory read mentions binary output."""
        self.expect(
            "help memory read",
            patterns=[
                "--show-tags\n\s+Include memory tags in output "
                "\(does not apply to binary output\)."
            ],
        )
        self.expect(
            "help memory find",
            patterns=["--show-tags\n\s+Include memory tags in output."],
        )

    @no_debug_info_test
    def test_help_show_enum_values(self):
        """Check the help output for a argument type contains the enum values
        and their descriptions."""
        self.expect(
            "help <log-handler>",
            substrs=[
                "The log handle that will be used to write out log messages.",
                "default",
                "Use the default (stream) log handler",
                "stream",
                "Write log messages to the debugger output stream",
                "circular",
                "Write log messages to a fixed size circular buffer",
                "os",
                "Write log messages to the operating system log",
            ],
        )
