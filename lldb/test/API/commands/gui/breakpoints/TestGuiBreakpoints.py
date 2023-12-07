"""
Test the 'gui' shortcut 'b' (toggle breakpoint).
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest


class TestGuiBasicDebugCommandTest(PExpectTest):
    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIfCursesSupportMissing
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    def test_gui(self):
        self.build()

        self.launch(executable=self.getBuildArtifact("a.out"), dimensions=(100, 500))
        self.expect(
            'br set -o true -f main.c -p "// First break here"',
            substrs=["Breakpoint 1", "address ="],
        )
        self.expect("run", substrs=["stop reason ="])

        self.child.sendline("breakpoint list")
        self.child.expect_exact("No breakpoints currently set.")

        escape_key = chr(27).encode()
        down_key = chr(27) + "OB"  # for vt100 terminal (lldbexpect sets TERM=vt100)

        # Start the GUI.
        self.child.sendline("gui")
        self.child.expect_exact("Sources")  # wait for gui

        # Go to next line, set a breakpoint.
        self.child.send(down_key)
        self.child.send("b")
        self.child.send(escape_key)
        self.expect_prompt()
        self.child.sendline("breakpoint list")
        self.child.expect("2: file = '[^']*main.c', line = 3,.*")
        self.child.sendline("gui")
        self.child.expect_exact("Sources")

        # Go two lines down ("gui" resets position), set a breakpoint.
        self.child.send(down_key)
        self.child.send(down_key)
        self.child.send("b")
        self.child.send(escape_key)
        self.expect_prompt()
        self.child.sendline("breakpoint list")
        self.child.expect("2: file = '[^']*main.c', line = 3,")
        self.child.expect("3: file = '[^']*main.c', line = 4,")
        self.child.sendline("gui")
        self.child.expect_exact("Sources")

        # Toggle both the breakpoints (remove them).
        self.child.send(down_key)
        self.child.send("b")
        self.child.send(down_key)
        self.child.send("b")
        self.child.send(escape_key)
        self.expect_prompt()
        self.child.sendline("breakpoint list")
        self.child.expect_exact("No breakpoints currently set.")
        self.child.sendline("gui")
        self.child.expect_exact("Sources")

        # Press escape to quit the gui
        self.child.send(escape_key)

        self.expect_prompt()
        self.quit()
