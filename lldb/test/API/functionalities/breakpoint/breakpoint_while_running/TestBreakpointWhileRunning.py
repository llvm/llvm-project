"""
Test inserting a breakpoint while inferior is executing.
"""

import lldb
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test import lldbutil


class BreakpointWhileRunning(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_breakpoint_while_running(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "main", lldb.SBFileSpec("main.c")
        )

        self.dbg.SetAsync(True)
        listener = self.dbg.GetListener()
        self.runCmd("break delete --force")
        self.runCmd("continue")
        lldbutil.expect_state_changes(self, listener, process, [lldb.eStateRunning])
        self.runCmd("break set --source-pattern-regexp 'break here'")
        lldbutil.expect_state_changes(self, listener, process, [lldb.eStateStopped])
