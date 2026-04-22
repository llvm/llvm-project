"""
Regression test for a bug in the default event handler (specifically when
redrawing the statusline) that triggered when auto-continuing from a
breakpoint.
"""

import os

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest


@skipIfTargetDoesNotSupportThreads()
@skipIfAsan
@skipIfEditlineSupportMissing
class BreakpointCommandAutoContinueTestCase(PExpectTest):
    NO_DEBUG_INFO_TESTCASE = True

    def test_breakpoint_command_auto_continue(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        bpcmd = os.path.join(self.getSourceDir(), "bpcmd.py")

        self.launch(executable=exe, timeout=60, dimensions=(25, 80))

        self.expect("breakpoint set --name break_here", substrs=["Breakpoint 1"])
        self.expect(
            f"command script import {bpcmd}",
        )
        self.expect(
            "breakpoint command add --python-function bpcmd.write_ok 1",
        )

        # Run the program. It should complete successfully (print PASSED).
        # Without the fix, the debugger would interrupt the process when
        # processing auto-continue events to fetch stale thread/frame state
        # for the statusline, causing the memory writes to fail and the
        # program to abort.
        self.child.sendline("run")
        self.child.expect("PASSED", timeout=30)
