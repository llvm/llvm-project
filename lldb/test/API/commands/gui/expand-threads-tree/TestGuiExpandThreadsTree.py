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

        print(1)
        self.launch(executable=self.getBuildArtifact("a.out"), dimensions=(100,500))
        print(2)
        self.expect("breakpoint set -n break_here", substrs=["Breakpoint 1", "address ="])
        print(3)
        self.expect("run", substrs=["stop reason ="])

        escape_key = chr(27).encode()

        # Start the GUI and close the welcome window.
        print(4)
        self.child.sendline("gui")
        print(5)
        self.child.send(escape_key)
        print(6)
        self.child.expect_exact("Threads")

        # The thread running thread_start_routine should be expanded.

        print(7)
        self.child.expect_exact("#0: break_here")

        # Exit GUI.

        print(8)
        self.child.send(escape_key)
        print(9)
        self.expect_prompt()

        # Select the main thread.
        print(10)
        self.child.sendline("thread select 1")

        # Start the GUI.
        print(11)
        self.child.sendline("gui")
        print(12)
        self.child.expect_exact("Threads")

        # The main thread should be expanded.

        print(13)
        self.child.expect("#\d+: main")

        # Quit the GUI
        print(14)
        self.child.send(escape_key)

        print(15)
        self.expect_prompt()

        print(16)
        self.quit()
