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

        # Intermediate breakpoint after stdout/stderr output but before large
        # output. This ensures those lines are visible in the console window
        # (only 6 lines exist) when the process pauses.
        self.expect(
            'br set -o true -f main.cpp -p "// break here middle"',
            substrs=["Breakpoint 2", "address ="],
        )

        self.expect(
            'br set -o true -f main.cpp -p "// break here end"',
            substrs=["Breakpoint 3", "address ="],
        )

        self.expect("run", substrs=["stop reason ="])

        escape_key = chr(27).encode()

        # Start the GUI.
        self.child.sendline("gui")

        # Check for gui elements in Menu bar (top of screen)
        self.child.expect_exact("Target")
        self.child.expect_exact("Process")
        self.child.expect_exact("View")

        # Check for window titles (middle of screen)
        self.child.expect_exact("Sources")
        self.child.expect_exact("Console")

        # The Console window shows this message before any output.
        self.child.expect_exact("(no output yet)")

        # Continue to the intermediate breakpoint (after generate_output()).
        self.child.send("c")

        # The GUI draws the Threads pane (which shows "stop reason") BEFORE
        # the Console pane in the byte stream. Check stop reason first so it
        # is not skipped over when scanning for the console text below.
        self.child.expect_exact("stop reason")

        # The console pane is drawn after the threads pane in the same redraw
        # cycle, so these lines appear later in the byte stream.
        # Only 6 lines of output exist at this stop, so all are visible.
        self.child.expect_exact("Hello from stdout line 1")
        self.child.expect_exact("Hello from stderr line 3")

        # Continue past the large output to the final breakpoint.
        self.child.send("c")

        # Again check stop reason before console output for the same reason.
        self.child.expect_exact("stop reason")

        # Large output lines near the end are guaranteed visible in the
        # auto-scrolled console window.
        self.child.expect_exact("Large output line 99")

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

        # Intermediate breakpoint after stdout/stderr, before large output.
        self.expect(
            'br set -o true -f main.cpp -p "// break here middle"',
            substrs=["Breakpoint 2", "address ="],
        )

        self.expect(
            'br set -o true -f main.cpp -p "// break here end"',
            substrs=["Breakpoint 3", "address ="],
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

        # The Console window shows this message before any output.
        self.child.expect_exact("(no output yet)")

        # Continue to the intermediate breakpoint (after generate_output()).
        self.child.send("c")

        # Threads pane ("stop reason") is drawn before the Console pane, so
        # check it first to avoid consuming it while scanning for console text.
        self.child.expect_exact("stop reason")

        # Check that stdout output is visible (only 6 lines at this stop).
        self.child.expect_exact("Hello from stdout line 1")

        # Tab to console window (Sources -> Threads -> Variables -> Console)
        self.child.send(tab_key)
        self.child.send(tab_key)
        self.child.send(tab_key)

        # Clear Console output ('c' in the Console window clears output)
        self.child.send("c")

        # The Console window shows this message after clear
        self.child.expect_exact("(no output yet)")

        # Press escape to quit the gui
        self.child.send(escape_key)

        self.expect_prompt()
        self.quit()
