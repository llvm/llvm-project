"""
Test that the lldb editline handling is configured correctly.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbpexpect import PExpectTest


class EditlineTest(PExpectTest):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_left_right_arrow(self):
        """Test that ctrl+left/right arrow navigates words correctly.

        Note: just sending escape characters to pexpect and checking the buffer
        doesn't work well, so we run real commands. We want to type
        "help command" while exercising word-navigation, so type it as below,
        where [] indicates cursor position.

        1. Send "el rint"  -> "el rint[]"
        2. Ctrl+left once  -> "el []rint"
        3. Send "p"        -> "el p[]rint"
        4. Ctrl+left twice -> "[]el print"
        5. Send "h"        -> "h[]el print"
        6. Ctrl+right      -> "hel[] print"
        7. Send "p"        -> "help print"
        """
        self.launch()

        escape_pairs = [
            ("\x1b[1;5D", "\x1b[1;5C"),
            ("\x1b[5D", "\x1b[5C"),
            ("\x1b\x1b[D", "\x1b\x1b[C"),
        ]
        for (l_escape, r_escape) in escape_pairs:
            self.expect("el rint{L}p{L}{L}h{R}p".format(
                L=l_escape, R=r_escape), substrs=["Syntax: print"])

        self.quit()
