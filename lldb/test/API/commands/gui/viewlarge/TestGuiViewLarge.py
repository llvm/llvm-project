"""
Test that the 'gui' displays long lines/names correctly without overruns.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest

class GuiViewLargeCommandTest(PExpectTest):

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIfCursesSupportMissing
    @skipIfRemote # "run" command will not work correctly for remote debug
    @expectedFailureNetBSD
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    def test_gui(self):
        self.build()

        # Limit columns to 80, so that long lines will not be displayed completely.
        self.launch(executable=self.getBuildArtifact("a.out"), dimensions=(100,80))
        self.expect('br set -f main.c -p "// Break here"', substrs=["Breakpoint 1", "address ="])
        self.expect("run", substrs=["stop reason ="])

        escape_key = chr(27).encode()
        left_key = chr(27)+'OD' # for vt100 terminal (lldbexpect sets TERM=vt100)
        right_key = chr(27)+'OC'
        ctrl_l = chr(12)

        # Start the GUI.
        self.child.sendline("gui")

        # Check the sources window.
        self.child.expect_exact("Sources")
        # The string is copy&pasted from a 80-columns terminal. It will be followed by some
        # kind of an escape sequence (color, frame, etc.).
        self.child.expect_exact("int a_variable_with_a_very_looooooooooooooooooooooooooo"+chr(27))
        # The escape here checks that there's no content drawn by the previous line.
        self.child.expect_exact("int shortvar = 1;"+chr(27))
        # Check the triggered breakpoint marker on a long line.
        self.child.expect_exact("<<< Thread 1: breakpoint 1.1"+chr(27))

        # Check the variable window.
        self.child.expect_exact("Variables")
        self.child.expect_exact("(int) a_variable_with_a_very_looooooooooooooooooooooooooooooo"+chr(27))
        self.child.expect_exact("(int) shortvar = 1"+chr(27))

        # Scroll the sources view twice to the right.
        self.child.send(right_key)
        self.child.send(right_key)
        # Force a redraw, otherwise curses will optimize the drawing to not draw all 'o'.
        self.child.send(ctrl_l)
        # The source code is indented by two spaces, so there'll be just two extra 'o' on the right.
        self.child.expect_exact("int a_variable_with_a_very_looooooooooooooooooooooooooooo"+chr(27))

        # And scroll back to the left.
        self.child.send(left_key)
        self.child.send(left_key)
        self.child.send(ctrl_l)
        self.child.expect_exact("int a_variable_with_a_very_looooooooooooooooooooooooooo"+chr(27))

        # Press escape to quit the gui
        self.child.send(escape_key)

        self.expect_prompt()
        self.quit()
