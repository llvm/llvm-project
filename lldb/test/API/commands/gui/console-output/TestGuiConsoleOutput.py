"""
Test that the 'gui' console output pane displays stdout / stderr from the debugged process
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest


class TestGuiConsoleOutputTest(PExpectTest):
    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIfCursesSupportMissing
    def test_gui_console_output(self):
        """Test that console pane prints messages"""
        self.build()

        self.launch(
            executable=self.getBuildArtifact("a.out"),
            dimensions=(100, 500),
            run_under=["env", "TERM=xterm"],
        )

        self.expect(
            'br set -o true -f main.cpp -p "// break here begin"',
            substrs=["Breakpoint 1", "address ="],
        )

        self.expect(
            'br set -o true -f main.cpp -p "// break here end"',
            substrs=["Breakpoint 2", "address ="],
        )

        self.expect("run", substrs=["stop reason ="])

        escape_key = chr(27).encode()

        # Start the GUI.
        self.child.sendline("gui")

        # Check for gui elements in Menu bar (top of screen)
        # We expect these in the order they appear to avoid consumption issues
        self.child.expect_exact("Target")
        self.child.expect_exact("Process")
        self.child.expect_exact("View")

        # Check for window titles (middle of screen)
        self.child.expect_exact("Sources")
        self.child.expect_exact("Console")

        # The Console window show this message before continuing
        self.child.expect_exact("(no output yet)")

        # Continue program execution
        self.child.send("c")

        # Check console output for messages
        self.child.expect_exact("Hello from stdout line 1")
        self.child.expect_exact("Hello from stderr line 1")
        self.child.expect_exact("Hello from stdout line 2")
        self.child.expect_exact("Hello from stderr line 2")
        self.child.expect_exact("Hello from stdout line 3")
        self.child.expect_exact("Hello from stderr line 3")

        # Wait for Breakpoint 2
        self.child.expect_exact("stop reason")

        # Press escape to quit the gui
        self.child.send(escape_key)

        self.expect_prompt()
        self.quit()

    @skipIfAsan
    @skipIfCursesSupportMissing
    def test_gui_console_navigate(self):
        """Test that console pane navigation works"""
        self.build()

        self.launch(
            executable=self.getBuildArtifact("a.out"),
            dimensions=(100, 500),
            run_under=["env", "TERM=xterm"],
        )

        self.expect(
            'br set -o true -f main.cpp -p "// break here begin"',
            substrs=["Breakpoint 1", "address ="],
        )

        self.expect(
            'br set -o true -f main.cpp -p "// break here end"',
            substrs=["Breakpoint 2", "address ="],
        )

        self.expect("run", substrs=["stop reason ="])

        escape_key = chr(27).encode()
        tab_key = chr(9).encode()

        # Start the GUI.
        self.child.sendline("gui")

        # Match elements in top-to-bottom order
        self.child.expect_exact("Target")
        self.child.expect_exact("Sources")
        self.child.expect_exact("Console")

        # The Console window show this message before continuing
        self.child.expect_exact("(no output yet)")

        # Continue program execution
        self.child.send("c")

        # Check console output for messages
        self.child.expect_exact("Hello from stdout line 1")

        # Wait for Breakpoint 2
        self.child.expect_exact("stop reason")

        # Tab to console
        self.child.send(tab_key)  # Sources -> Threads
        self.child.send(tab_key)  # Threads -> Variables
        self.child.send(tab_key)  # Variables -> Console

        # Clear Console output
        self.child.send("c")

        # The Console window show this message after clear
        self.child.expect_exact("(no output yet)")

        # Press escape to quit the gui
        self.child.send(escape_key)

        self.expect_prompt()
        self.quit()
