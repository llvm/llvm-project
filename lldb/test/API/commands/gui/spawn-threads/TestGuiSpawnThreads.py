"""
Test that 'gui' does not crash when adding new threads, which
populate TreeItem's children and may be reallocated elsewhere.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest

import sys


class TestGuiSpawnThreadsTest(PExpectTest):
    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIfCursesSupportMissing
    def test_gui(self):
        self.build()

        self.launch(executable=self.getBuildArtifact("a.out"), dimensions=(100, 500))
        self.expect(
            'breakpoint set -f main.cpp -p "break here"',
            substrs=["Breakpoint 1", "address ="],
        )
        self.expect(
            'breakpoint set -f main.cpp -p "before join"',
            substrs=["Breakpoint 2", "address ="],
        )
        self.expect("run", substrs=["stop reason ="])

        escape_key = chr(27).encode()

        # Start the GUI
        self.child.sendline("gui")
        self.child.expect_exact("Threads")
        self.child.expect_exact(f"thread #1: tid =")

        for i in range(5):
            # Stopped at the breakpoint, continue over the thread creation
            self.child.send("c")
            # Check the newly created thread
            self.child.expect_exact(f"thread #{i + 2}: tid =")

        # Exit GUI.
        self.child.send(escape_key)
        self.expect_prompt()

        self.quit()
