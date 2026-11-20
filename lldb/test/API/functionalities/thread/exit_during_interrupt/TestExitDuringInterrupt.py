"""
Test interrupting a process while its threads exit.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@requireThreadSupport
class ExitDuringInterruptTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()
        target, process, _, _ = lldbutil.run_to_name_breakpoint(self, "main")
        target.DeleteAllBreakpoints()

        self.setAsync(True)
        listener = self.dbg.GetListener()

        self.runCmd("continue")
        lldbutil.expect_state_changes(self, listener, process, [lldb.eStateRunning])

        for _ in range(50):
            self.runCmd("process interrupt")
            lldbutil.expect_state_changes(self, listener, process, [lldb.eStateStopped])
            self.runCmd("continue")
            lldbutil.expect_state_changes(self, listener, process, [lldb.eStateRunning])
