"""
Test detaching from a process that lldb launched and that keeps running.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DetachLaunchedTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfRemote
    @skipIfWindows
    def test_detach_from_a_process_that_keeps_running(self):
        """lldb reports the detach although the process keeps its terminal open
        without writing to it. lldb stops reading the terminal before it
        reports the detach, and if that waited for the read to time out, the
        debug server would exit first, and lldb would report that the process
        exited."""
        self.build()
        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )
        self.assertSuccess(process.Detach())
        self.assertState(process.GetState(), lldb.eStateDetached)
