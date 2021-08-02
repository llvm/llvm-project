"""
Test the 'gui' shortcuts 's','n','f','u','d' (step in, step over, step out, up, down)
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest

class TestGuiBasicDebugCommandTest(PExpectTest):

    mydir = TestBase.compute_mydir(__file__)

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIf(oslist=["linux"], archs=["arm","aarch64"])
    @skipIfCursesSupportMissing
    def test_gui(self):
        self.build()

        self.launch(executable=self.getBuildArtifact("a.out"), dimensions=(100,500))
        self.expect('br set -f main.c -p "// Break here"', substrs=["Breakpoint 1", "address ="])
        self.expect("run", substrs=["stop reason ="])

        escape_key = chr(27).encode()

        # Start the GUI and close the welcome window.
        self.child.sendline("gui")
        self.child.expect("Welcome to the LLDB curses GUI.")
        self.child.send(escape_key)

        # Simulate a simple debugging session.
        self.child.send("s") # step
        self.child.expect("return 1; // In function[^\r\n]+<<< Thread 1: step in")
        self.child.send("u") # up
        self.child.expect_exact("func(); // Break here")
        self.child.send("d") # down
        self.child.expect_exact("return 1; // In function")
        self.child.send("f") # finish
        self.child.expect("<<< Thread 1: step out")
        self.child.send("s") # move onto the second one
        self.child.expect("<<< Thread 1: step in")
        self.child.send("n") # step over
        self.child.expect("<<< Thread 1: step over")

        # Press escape to quit the gui
        self.child.send(escape_key)

        self.expect_prompt()
        self.quit()
