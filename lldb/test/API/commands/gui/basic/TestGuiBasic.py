"""
Test that the 'gui' displays the help window and basic UI.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest


class BasicGuiCommandTest(PExpectTest):
    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIfCursesSupportMissing
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    def test_gui(self):
        self.build()

        self.launch(executable=self.getBuildArtifact("a.out"), dimensions=(100, 500))
        self.expect(
            'br set -f main.c -p "// Break here"', substrs=["Breakpoint 1", "address ="]
        )
        self.expect("run", substrs=["stop reason ="])

        escape_key = chr(27).encode()

        # Start the GUI.
        self.child.sendline("gui")

        # Check for GUI elements in the menu bar.
        self.child.expect_exact("Target")
        self.child.expect_exact("Process")
        self.child.expect_exact("Thread")
        self.child.expect_exact("View")
        self.child.expect_exact("Help")

        # Check the sources window.
        self.child.expect_exact("Sources")
        self.child.expect_exact("main")
        self.child.expect_exact("funky_var_name_that_should_be_rendered")

        # Check the variable window.
        self.child.expect_exact("Variables")
        self.child.expect_exact("(int) funky_var_name_that_should_be_rendered = 22")

        # Check the bar at the bottom.
        self.child.expect_exact("Frame:")

        # Press escape to quit the gui
        self.child.send(escape_key)

        self.expect_prompt()
        self.quit()
