"""
Test that the 'gui' displays long lines/names correctly without overruns.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest

class GuiViewLargeCommandTest(PExpectTest):

    mydir = TestBase.compute_mydir(__file__)

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIfCursesSupportMissing
    @skipIfRemote # "run" command will not work correctly for remote debug
    def test_gui(self):
        self.build()

        # Limit columns to 80, so that long lines will not be displayed completely.
        self.launch(executable=self.getBuildArtifact("a.out"), dimensions=(100,80))
        self.expect('br set -f main.c -p "// Break here"', substrs=["Breakpoint 1", "address ="])
        self.expect("run", substrs=["stop reason ="])

        escape_key = chr(27).encode()

        # Start the GUI and close the welcome window.
        self.child.sendline("gui")
        self.child.send(escape_key)

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

        # Press escape to quit the gui
        self.child.send(escape_key)

        self.expect_prompt()
        self.quit()
