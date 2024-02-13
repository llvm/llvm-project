"""
Make sure that the concurrent vfork() from multiple threads works correctly.
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestConcurrentVFork(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows
    def test_vfork_follow_parent(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        self.runCmd("settings set target.process.follow-fork-mode parent")
        self.expect("continue", substrs=["exited with status = 0"])

    @skipIfWindows
    def test_vfork_follow_child(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        self.runCmd("settings set target.process.follow-fork-mode child")
        self.expect("continue", substrs=["exited with status = 0"])
