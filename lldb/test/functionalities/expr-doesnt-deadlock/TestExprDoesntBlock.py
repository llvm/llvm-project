"""
Test that expr will time out and allow other threads to run if it blocks.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class ExprDoesntDeadlockTestCase(TestBase):

    def getCategories(self):
        return ['basic_process']

    mydir = os.path.join("functionalities", "expr-doesnt-deadlock")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test that expr will time out and allow other threads to run if it blocks - with dsym."""
        self.buildDsym()
        self.expr_doesnt_deadlock()

    @dwarf_test
    @skipIfLinux # llvm.org/pr15258: disabled due to assertion failure in ProcessMonitor::GetCrashReasonForSIGSEGV:
    def test_with_dwarf_and_run_command(self):
        """Test that expr will time out and allow other threads to run if it blocks."""
        self.buildDwarf()
        self.expr_doesnt_deadlock()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def expr_doesnt_deadlock (self):
        """Test that expr will time out and allow other threads to run if it blocks."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint at source line before call_me_to_get_lock gets called.

        main_file_spec = lldb.SBFileSpec ("locking.c")
        breakpoint = target.BreakpointCreateBySourceRegex('Break here', main_file_spec)
        if self.TraceOn():
            print "breakpoint:", breakpoint
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be on self.line1 and the break condition should hold.
        from lldbutil import get_stopped_thread
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint condition")

        frame0 = thread.GetFrameAtIndex(0)

        var = frame0.EvaluateExpression ("call_me_to_get_lock()")
        self.assertTrue (var.IsValid())
        self.assertTrue (var.GetValueAsSigned (0) == 567)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
