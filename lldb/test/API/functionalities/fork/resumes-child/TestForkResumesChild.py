"""
Make sure that the fork child keeps running.
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestForkResumesChild(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows
    def test_step_over_fork(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.cpp"))
        self.runCmd("next")
        self.expect("continue", substrs = ["exited with status = 0"])
