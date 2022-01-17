"""
This test verifies the correct handling of the situation when a thread exits
while another thread triggers the termination (exit) of the entire process.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class ConcurrentThreadExitTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(oslist=no_match(["linux"]))
    def test(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        self.expect("run", substrs=["exited with status = 47"])
