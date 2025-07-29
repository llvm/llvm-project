"""
Test that disabling breakpoints and viewing them in a list uses the correct ANSI color settings when colors are enabled and disabled.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbpexpect import PExpectTest

import io


class DisabledBreakpointsTest(PExpectTest):
    @add_test_categories(["pexpect"])
    def test_disabling_breakpoints_with_color(self):
        """Test that disabling a breakpoint and viewing the breakpoints list uses the specified ANSI color prefix."""
        ansi_red_color_code = "\x1b[31m"

        self.launch(use_colors=True, dimensions=(100, 100))
        self.expect('settings set disable-ansi-prefix "${ansi.fg.red}"')
        self.expect("b main")
        self.expect("br dis")
        self.expect("br l", substrs=[ansi_red_color_code + "1:"])
        self.quit()
