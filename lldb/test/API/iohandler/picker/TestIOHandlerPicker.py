"""
Tests for the IOHandlerPicker interactive widget.

`test-picker` lists items in descending id order (5, 4, 3, 2, 1 for a count of
5), and multi-select mode pre-selects ids 1, 4, 7, ...
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest


@skipIfAsan
@skipIfRemote
@skipIfWindows
class IOHandlerPickerTest(PExpectTest):
    arrow_down = "\x1b[B"
    escape_key = "\x1b"

    def start_picker(self, command, header, dimensions=(24, 120), setup=()):
        if not self.dbg.GetCommandInterpreter().CommandExists("test-picker"):
            self.skipTest("test-picker requires a build with assertions")
        self.launch(dimensions=dimensions)
        # launch() runs "settings clear --all", which resets the terminal size.
        height, width = dimensions
        self.expect(f"settings set term-height {height}")
        self.expect(f"settings set term-width {width}")
        for setup_command, patterns in setup:
            self.expect(setup_command, patterns)
        self.child.sendline(command)
        self.child.expect_exact(header)

    def finish(self, *expected):
        for line in expected:
            self.child.expect_exact(line)
        self.expect_prompt()
        self.quit()

    def test_single_select_arrows(self):
        """Navigate with arrow keys and select with Enter."""
        self.start_picker("test-picker single 5", "Select an item:")
        self.child.send(self.arrow_down)
        self.child.send(self.arrow_down)
        self.child.send("\r")
        self.finish("selected: 3")

    def test_single_select_vim_keys(self):
        """Navigate with j/k and select with Enter."""
        self.start_picker("test-picker single 5", "Select an item:")
        self.child.send("jjk")
        self.child.send("\r")
        self.finish("selected: 4")

    def test_home_end_tilde_sequences(self):
        """Home and End also work as "ESC [ 1 ~" and "ESC [ 4 ~"."""
        self.start_picker("test-picker single 5", "Select an item:")
        # Home, End, then up: the second-to-last item.
        self.child.send("jj\x1b[1~\x1b[4~k\r")
        self.finish("selected: 2")

    def test_single_select_cancel(self):
        """ESC cancels without selecting anything."""
        self.start_picker("test-picker single 5", "Select an item:")
        self.child.send(self.escape_key)
        self.finish("canceled")

    def test_multi_select_toggle(self):
        """Space toggles items and Enter confirms the checked set."""
        self.start_picker("test-picker multi 5", "Toggle items:")
        # Check 5, then uncheck the pre-selected 4. Pre-selected 1 stays.
        self.child.send(" j ")
        self.child.send("\r")
        self.finish("selected: 5", "selected: 1")

    def test_multi_select_cancel(self):
        """ESC discards toggles."""
        self.start_picker("test-picker multi 5", "Toggle items:")
        self.child.send(" j ")
        self.child.send(self.escape_key)
        self.finish("canceled")

    def test_small_terminal_scrolls(self):
        """Moving past the viewport scrolls the list."""
        self.start_picker(
            "test-picker single 20", "Select an item:", dimensions=(10, 60)
        )
        self.child.send("j" * 6)
        self.child.expect_exact("[more above]")
        self.child.send("\r")
        self.finish("selected: 14")

    def test_dialog_centered(self):
        """The dialog is drawn in the middle of the screen."""
        self.start_picker("test-picker single 5", "Select an item:")
        # The dialog is 11 lines high, with (24 - 11) / 2 lines above it, and
        # leaves a tenth of the terminal's width on each side.
        self.child.send("j")
        self.child.expect_exact("\x1b[7;13H")
        self.child.send("\r")
        self.finish("selected: 4")

    def test_restores_screen(self):
        """Closing the dialog redraws what LLDB printed under it."""
        self.start_picker(
            "test-picker single 5",
            "Select an item:",
            setup=[("help test-picker", ["Test the IOHandlerPicker"])],
        )
        self.child.send("\r")
        # The earlier command and its output, then the picker's command.
        self.finish(
            "help test-picker",
            "Test the IOHandlerPicker",
            "test-picker single 5",
            "selected: 5",
        )

    def test_resize_narrower(self):
        """Shrinking the terminal's width clips the dialog to the new width."""
        self.start_picker("test-picker single 5", "Select an item:")
        self.child.setwinsize(24, 30)
        # The content is wider than the terminal, so the dialog drops its
        # margins, and clips the key help to the 28 columns inside its padding.
        self.child.expect_exact("  jk: navigate  Enter: selec \x1b[0m")
        self.child.send("j\r")
        self.finish("selected: 4")

    def test_resize_shorter(self):
        """Shrinking the terminal's height shrinks the list's viewport."""
        self.start_picker("test-picker single 20", "Select an item:")
        self.child.setwinsize(10, 120)
        # Ten lines fit the dialog's six lines of chrome and two rows, with a
        # line of the screen visible above and below it.
        self.child.expect_exact("Select an item:")
        self.child.send("j" * 4)
        self.child.expect_exact("[more above]")
        self.child.send("\r")
        self.finish("selected: 16")

    def test_statusline_stays(self):
        """The dialog draws above the statusline instead of removing it."""
        self.start_picker(
            "test-picker single 5",
            "Select an item:",
            setup=[("settings set show-statusline true", ["\x1b[1;23r"])],
        )
        self.child.send("j\r")
        self.child.expect_exact("selected: 4")
        # Resetting the scroll region to the whole terminal would remove the
        # statusline.
        self.assertNotIn(b"\x1b[1;24r", self.child.before)
        self.expect_prompt()
        self.quit()
