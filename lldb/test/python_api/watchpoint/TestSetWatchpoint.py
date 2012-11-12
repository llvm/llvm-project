"""
Use lldb Python SBValue API to create a watchpoint for read_write of 'globl' var.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class SetWatchpointAPITestCase(TestBase):

    mydir = os.path.join("python_api", "watchpoint")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.c'
        # Find the line number to break inside main().
        self.line = line_number(self.source, '// Set break point at this line.')

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_watch_val_with_dsym(self):
        """Exercise SBValue.Watch() API to set a watchpoint."""
        self.buildDsym()
        self.do_set_watchpoint()

    @python_api_test
    @dwarf_test
    def test_watch_val_with_dwarf(self):
        """Exercise SBValue.Watch() API to set a watchpoint."""
        self.buildDwarf()
        self.do_set_watchpoint()

    def do_set_watchpoint(self):
        """Use SBFrame.WatchValue() to set a watchpoint and verify that the program stops later due to the watchpoint."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c.
        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        # We should be stopped due to the breakpoint.  Get frame #0.
        process = target.GetProcess()
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        frame0 = thread.GetFrameAtIndex(0)

        # Watch 'global' for read and write.
        value = frame0.FindValue('global', lldb.eValueTypeVariableGlobal)
        error = lldb.SBError();
        watchpoint = value.Watch(True, True, True, error)
        self.assertTrue(value and watchpoint,
                        "Successfully found the variable and set a watchpoint")
        self.DebugSBValue(value)

        # Hide stdout if not running with '-t' option.
        if not self.TraceOn():
            self.HideStdout()

        print watchpoint

        # Continue.  Expect the program to stop due to the variable being written to.
        process.Continue()

        if (self.TraceOn()):
            lldbutil.print_stacktraces(process)

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonWatchpoint)
        self.assertTrue(thread, "The thread stopped due to watchpoint")
        self.DebugSBValue(value)

        # Continue.  Expect the program to stop due to the variable being read from.
        process.Continue()

        if (self.TraceOn()):
            lldbutil.print_stacktraces(process)

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonWatchpoint)
        self.assertTrue(thread, "The thread stopped due to watchpoint")
        self.DebugSBValue(value)

        # Continue the process.  We don't expect the program to be stopped again.
        process.Continue()

        # At this point, the inferior process should have exited.
        self.assertTrue(process.GetState() == lldb.eStateExited, PROCESS_EXITED)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
