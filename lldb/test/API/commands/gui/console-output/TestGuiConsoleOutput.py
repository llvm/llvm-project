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
    @skipIf(oslist=["linux"], archs=["arm$", "aarch64"])
    def test_gui_console_output(self):
        """Test that console pane prints messages"""
        self.build()

        self.launch(executable=self.getBuildArtifact("a.out"), dimensions=(100, 500))
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
        self.child.expect_exact("Sources")  # wait for gui

        # Check for other gui elements in Menu bar
        self.child.expect_exact("Target")
        self.child.expect_exact("Process")
        self.child.expect_exact("View")

        # Check Console window exists
        self.child.expect_exact("Console")

        # The Console window show this message before continuing
        self.child.expect_exact("(no output yet)")

        # Continue program execution
        self.child.send('c')

        # Wait for Breakpoint 2
        self.child.expect_exact("stop reason")

        # Check console output for messages
        self.child.expect_exact("Hello from stdout line 1")
        self.child.expect_exact("Hello from stderr line 1")
        self.child.expect_exact("Hello from stdout line 2")
        self.child.expect_exact("Hello from stderr line 2")
        self.child.expect_exact("Hello from stdout line 3")
        self.child.expect_exact("Hello from stderr line 3")

        # Press escape to quit the gui
        self.child.send(escape_key)

        self.expect_prompt()
        self.quit()

    @skipIfAsan
    @skipIfCursesSupportMissing
    @skipIf(oslist=["linux"], archs=["arm$", "aarch64"])
    def test_gui_console_menu_toggle(self):
        """Test that console pane can be toggled via View Menu"""
        self.build()

        self.launch(executable=self.getBuildArtifact("a.out"), dimensions=(100, 500))
        self.expect(
            'br set -o true -f main.cpp -p "// break here begin"',
            substrs=["Breakpoint 1", "address ="],
        )
        self.expect("run", substrs=["stop reason ="])

        escape_key = chr(27).encode()

        # Start the GUI.
        self.child.sendline("gui")
        self.child.expect_exact("Sources")  # wait for gui

        # Check Console window exists by default
        self.child.expect_exact("Console")

        # Open View Menu and toggle Console window off
        self.child.send('v')
        self.child.expect_exact("Console") # menu item should exist
        self.child.send('o')

        # Wait for gui update
        import time
        time.sleep(0.5)

        # Open View Menu and toggle Console window on
        self.child.send('v')
        self.child.expect_exact("Console") # menu item should exist
        self.child.send('o')

        # Console window show re-appear
        self.child.expect_exact("Console")

        # Press escape to quit the gui
        self.child.send(escape_key)

        self.expect_prompt()
        self.quit()

    @skipIfAsan
    @skipIfCursesSupportMissing
    @skipIf(oslist=["linux"], archs=["arm$", "aarch64"])
    def test_gui_console_navigate(self):
        """Test that console pane navigation works"""
        self.build()

        self.launch(executable=self.getBuildArtifact("a.out"), dimensions=(100, 500))
        self.expect(
            'br set -o true -f main.cpp -p "// break here first"',
            substrs=["Breakpoint 1", "address ="],
        )
        self.expect(
            'br set -o true -f main.cpp -p "// break here end"',
            substrs=["Breakpoint 2", "address ="],
        )
        self.expect("run", substrs=["stop reason ="])

        escape_key = chr(27).encode()
        tab_key    = chr(9).encode()

        # Start the GUI.
        self.child.sendline("gui")
        self.child.expect_exact("Sources")  # wait for gui

        # Check Console window exists by default
        self.child.expect_exact("Console")

        # The Console window show this message before continuing
        self.child.expect_exact("(no output yet)")

        # Continue program execution
        self.child.send('c')

        # Wait for Breakpoint 2
        self.child.expect_exact("stop reason")

        # Check console output for messages
        self.child.expect_exact("Hello from stdout line 1")

        # Tab to console
        self.child.send(tab_key) # Sources -> Threads
        self.child.send(tab_key) # Threads -> Variables
        self.child.send(tab_key) # Variables -> Console

        # Clear Console output
        self.child.send('c')

        # The Console window show this message after clear
        self.child.expect_exact("(no output yet)")

        # Press escape to quit the gui
        self.child.send(escape_key)

        self.expect_prompt()
        self.quit()

    @skipIfAsan
    @skipIfCursesSupportMissing
    @skipIf(oslist=["linux"], archs=["arm$", "aarch64"])
    def test_gui_console_interaction(self):
        """Test that console pane doesn't interfere with other window layouts"""
        self.build()

        self.launch(executable=self.getBuildArtifact("a.out"), dimensions=(100, 500))
        self.expect(
            'br set -o true -f main.cpp -p "// break here begin"',
            substrs=["Breakpoint 1", "address ="],
        )
        self.expect("run", substrs=["stop reason ="])

        escape_key = chr(27).encode()

        # Start the GUI.
        self.child.sendline("gui")
        self.child.expect_exact("Sources")  # wait for gui

        # Check Console window exists by default
        self.child.expect_exact("Console")

        # Check other windows exists
        self.child.expect_exact("Threads")
        self.child.expect_exact("Variables")

        # Check test_var variable is listed in Variables window
        self.child.expect_exact("test_var")

        # Check source code in shown Sources window
        self.child.expect_exact("main.cpp")

        # Check main thread is shown in Threads window
        self.child.expect_exact("thread #1")

        # Press escape to quit the gui
        self.child.send(escape_key)

        self.expect_prompt()
        self.quit()
