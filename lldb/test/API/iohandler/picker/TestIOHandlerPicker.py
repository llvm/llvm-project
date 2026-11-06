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

    def start_picker(self, command, header, dimensions=(24, 120)):
        # `test-picker` is only registered in builds with assertions enabled.
        if command.startswith("test-picker") and not (
            self.dbg.GetCommandInterpreter().CommandExists("test-picker")
        ):
            self.skipTest("test-picker requires a build with assertions")
        self.launch(dimensions=dimensions)
        # launch() runs "settings clear --all", which resets the terminal size.
        height, width = dimensions
        self.expect(f"settings set term-height {height}")
        self.expect(f"settings set term-width {width}")
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

    def test_sort_ascending(self):
        """The first s sorts by the first column, ascending."""
        self.start_picker("test-picker single 5", "Select an item:")
        self.child.send("s")
        self.child.expect_exact("ID ^")
        self.child.send("j\r")
        self.finish("selected: 2")

    def test_sort_cycle(self):
        """Each s advances the sort: ID asc, ID desc, NAME asc, NAME desc."""
        self.start_picker("test-picker single 5", "Select an item:")
        self.child.send("ssss")
        self.child.expect_exact("NAME v")
        # NAME descending: echo (5), delta (4), ...
        self.child.send("j\r")
        self.finish("selected: 4")

    def test_small_terminal_scrolls(self):
        """Moving past the viewport scrolls the list."""
        self.start_picker(
            "test-picker single 20", "Select an item:", dimensions=(10, 60)
        )
        self.child.send("j" * 6)
        self.child.expect_exact("[more above]")
        self.child.send("\r")
        self.finish("selected: 14")

    def test_search_filters_rows(self):
        """/ filters rows to those with a column containing the query."""
        self.start_picker("test-picker single 26", "Select an item:")
        # Matches quebec (17) and echo (5). Enter applies the filter.
        self.child.send("/ec\r")
        self.child.expect_exact("filter: ec  (24 hidden)")
        self.child.send("j\r")
        self.finish("selected: 5")

    def test_search_typing_is_not_navigation(self):
        """While searching, j/k/s/Space are part of the query and case is
        ignored."""
        self.start_picker("test-picker single 26", "Select an item:")
        self.child.send("/jULIET\r\r")
        self.finish("selected: 10")

    def test_search_backspace(self):
        """Backspace edits the query and brings back matches."""
        self.start_picker("test-picker single 26", "Select an item:")
        self.child.send("/ecx")
        self.child.expect_exact("no matches  (26 hidden)")
        self.child.send("\x7f\r\r")
        self.finish("selected: 17")

    def test_search_ignores_unhandled_sequences(self):
        """Ignored escape sequences don't edit the query."""
        self.start_picker("test-picker single 26", "Select an item:")
        # Delete is "ESC [ 3 ~".
        self.child.send("/ec\x1b[3~\r")
        self.child.expect_exact("filter: ec  (24 hidden)")
        self.child.send("j\r")
        self.finish("selected: 5")

    def test_search_escape_clears(self):
        """ESC while searching clears the filter instead of canceling."""
        self.start_picker("test-picker single 26", "Select an item:")
        self.child.send("/zz")
        self.child.expect_exact("/zz_")
        self.child.send(self.escape_key)
        self.child.expect_exact("/: search")
        self.child.send("\r")
        self.finish("selected: 26")

    def test_search_multi_keeps_hidden_selection(self):
        """Items hidden by the filter stay checked in multi-select mode."""
        self.start_picker("test-picker multi 5", "Toggle items:")
        # Matches echo (5) and charlie (3); check 5. 4 and 1 are hidden but
        # stay pre-selected.
        self.child.send("/ch\r \r")
        self.finish("selected: 5", "selected: 4", "selected: 1")

    def test_process_attach_picker(self):
        """`process attach` without a pid, name, or target opens the picker."""
        self.start_picker("process attach", "Select a process to attach to:")
        self.child.send(self.escape_key)
        self.finish("no process selected")
