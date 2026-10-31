"""
Tests autosuggestion using pexpect.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest


def cursor_horizontal_abs(s):
    return "\x1b[" + str(len(s) + 1) + "G"


class TestCase(PExpectTest):
    ANSI_FAINT = "\x1b[2m"
    ANSI_RESET = "\x1b[0m"
    ANSI_RED = "\x1b[31m"
    ANSI_CYAN = "\x1b[36m"

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_autosuggestion_add_spaces(self):
        self.launch(
            use_colors=True,
            extra_args=[
                "-o",
                "settings set show-autosuggestion true",
                "-o",
                "settings set use-color true",
            ],
        )

        # Check if spaces are added to hide the previous gray characters.
        self.expect("help frame var")
        self.expect("help frame info")
        self.child.send("help frame v")
        self.child.expect_exact(
            cursor_horizontal_abs("(lldb) help frame ")
            + "v"
            + self.ANSI_FAINT
            + "ar"
            + self.ANSI_RESET
            + " "
        )


    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_autosuggestion(self):
        self.launch(
            use_colors=True,
            extra_args=[
                "-o",
                "settings set show-autosuggestion true",
                "-o",
                "settings set use-color true",
            ],
        )

        # Common input codes.
        ctrl_f = "\x06"
        delete = chr(127)

        frame_output_needle = "Syntax: frame <subcommand>"
        # Run 'help frame' once to put it into the command history.
        self.expect("help frame", substrs=[frame_output_needle])

        # Check that LLDB shows the autosuggestion in gray behind the text.
        self.child.send("hel")
        self.child.expect_exact(
            cursor_horizontal_abs("(lldb) he")
            + "l"
            + self.ANSI_FAINT
            + "p frame"
            + self.ANSI_RESET
        )

        # Apply the autosuggestion and press enter. This should print the
        # 'help frame' output if everything went correctly.
        self.child.send(ctrl_f + "\n")
        self.child.expect_exact(frame_output_needle)

        # Check that pressing Ctrl+F directly after Ctrl+F again does nothing.
        self.child.send("hel" + ctrl_f + ctrl_f + "\n")
        self.child.expect_exact(frame_output_needle)

        # Try autosuggestion using tab and ^f.
        # \t makes "help" and ^f makes "help frame". If everything went
        # correct we should see the 'help frame' output again.
        self.child.send("hel\t" + ctrl_f + "\n")
        self.child.expect_exact(frame_output_needle)

        # Check that autosuggestion works after delete.
        self.child.send("a1234" + 5 * delete + "hel" + ctrl_f + "\n")
        self.child.expect_exact(frame_output_needle)

        # Check that autosuggestion works after delete.
        self.child.send("help x" + delete + ctrl_f + "\n")
        self.child.expect_exact(frame_output_needle)

        # Check that autosuggestion complete to the most recent one.
        self.child.send("help frame variable\n")
        self.child.send("help fr")
        self.child.expect_exact(self.ANSI_FAINT + "ame variable" + self.ANSI_RESET)
        self.child.send("\n")

        # Try another command.
        apropos_output_needle = "Syntax: apropos <search-word>"
        # Run 'help frame' once to put it into the command history.
        self.expect("help apropos", substrs=[apropos_output_needle])

        # Check that 'hel' should have an autosuggestion for 'help apropos' now.
        self.child.send("hel")
        self.child.expect_exact(
            cursor_horizontal_abs("(lldb) he")
            + "l"
            + self.ANSI_FAINT
            + "p apropos"
            + self.ANSI_RESET
        )

        # Run the command and expect the 'help apropos' output.
        self.child.send(ctrl_f + "\n")
        self.child.expect_exact(apropos_output_needle)

        # Check that pressing Ctrl+F in an empty prompt does nothing.
        breakpoint_output_needle = "Syntax: breakpoint <subcommand>"
        self.child.send(ctrl_f + "help breakpoint" + "\n")
        self.child.expect_exact(breakpoint_output_needle)


    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_autosuggestion_custom_ansi_prefix_suffix(self):
        self.launch(
            use_colors=True,
            extra_args=[
                "-o",
                "settings set show-autosuggestion true",
                "-o",
                "settings set use-color true",
                "-o",
                "settings set show-autosuggestion-ansi-prefix ${ansi.fg.red}",
                "-o",
                "setting set show-autosuggestion-ansi-suffix ${ansi.fg.cyan}",
            ],
        )

        self.child.send("help frame variable\n")
        self.child.send("help fr")
        self.child.expect_exact(self.ANSI_RED + "ame variable" + self.ANSI_CYAN)
        self.child.send("\n")

    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_autosuggestion_tab_mode(self):
        """Test that show-autosuggestion=tab-mode suggests the prefix that
        tab completion would insert, instead of consulting command history."""
        self.launch(
            use_colors=True,
            extra_args=[
                "-o",
                "settings set show-autosuggestion tab-mode",
                "-o",
                "settings set use-color true",
            ],
        )

        ctrl_f = "\x06"

        # Put 'help frame' into the session history. If tab-mode incorrectly
        # consulted history (as the 'true' mode does), typing 'hel' would
        # suggest 'p frame'. In tab-mode the suggestion should be just 'p',
        # since the only tab completion of 'hel' is 'help'.
        frame_output_needle = "Syntax: frame <subcommand>"
        self.expect("help frame", substrs=[frame_output_needle])

        self.child.send("hel")
        # Note that the ANSI_RESET color prevents us from accepting a longer
        # suggestion as a valid test outcome.
        self.child.expect_exact(
            cursor_horizontal_abs("(lldb) he")
            + "l"
            + self.ANSI_FAINT
            + "p"
            + self.ANSI_RESET
        )

        # Applying the suggestion with Ctrl-F should leave the line as 'help'
        # (not 'help frame'). Running it lists all commands.
        self.child.send(ctrl_f + "\n")
        self.child.expect_exact("Debugger commands:")

        # Tab completion must still work in tab-mode. Pressing tab on 'hel'
        # should complete to 'help'; running it lists all commands.
        self.child.send("hel\t\n")
        self.child.expect_exact("Debugger commands:")

    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_autosuggestion_tab_mode_no_common_prefix(self):
        """Test that show-autosuggestion=tab-mode shows no suggestion when
        the tab completions don't share a prefix beyond what the user typed."""
        self.launch(
            use_colors=True,
            extra_args=[
                "-o",
                "settings set show-autosuggestion tab-mode",
                "-o",
                "settings set use-color true",
            ],
        )

        ctrl_f = "\x06"

        # 'se' matches several top-level commands (session, settings, ...)
        # whose longest common prefix is 'se' itself, so tab-mode must not
        # show any suggestion. Pressing Ctrl-F should therefore do nothing
        # and running the line should yield the 'ambiguous command' error.
        self.child.send("se" + ctrl_f + "\n")
        self.child.expect_exact("ambiguous command 'se'")

        # Tab completion must still work in tab-mode even when there is no
        # common prefix to suggest: pressing tab on 'se' should list the
        # available completions.
        self.child.send("se\t")
        self.child.expect_exact("session")
        self.child.expect_exact("settings")
