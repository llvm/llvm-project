"""
Test the 'gui' default thread tree expansion.
The root process tree item and the tree item corresponding to the selected
thread should be expanded by default.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest

class TestGuiExpandThreadsTree(PExpectTest):

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIfCursesSupportMissing
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    @skipIf(bugnumber="rdar://97460266")
    def test_gui(self):
        self.build()

        self.launch(executable=self.getBuildArtifact("a.out"), dimensions=(100,500))
        self.expect("breakpoint set -n break_here", substrs=["Breakpoint 1", "address ="])
        self.expect("run", substrs=["stop reason ="])

        escape_key = chr(27).encode()

        # Start the GUI and close the welcome window.
        self.child.sendline("gui")
        self.child.send(escape_key)
        self.child.expect_exact("Threads")

        # The thread running thread_start_routine should be expanded.
        self.child.expect_exact("#0: break_here")

        # Exit GUI.
        self.child.send(escape_key)
        self.expect_prompt()

        # Select the main thread.
        self.child.sendline("thread select 1")

        # Start the GUI.
        self.child.sendline("gui")
        self.child.expect_exact("Threads")

        # The main thread should be expanded.
        self.child.expect("#\d+: main")

        # Quit the GUI
        self.child.send(escape_key)

        self.expect_prompt()
        self.quit()
