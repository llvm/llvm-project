"""
Test watchpoint condition API.
"""

import os, time
import unittest2
import lldb
import lldbutil
from lldbtest import *

class WatchpointConditionAPITestCase(TestBase):

    mydir = os.path.join("python_api", "watchpoint", "condition")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.cpp'
        # Find the line number to break inside main().
        self.line = line_number(self.source, '// Set break point at this line.')
        # And the watchpoint variable declaration line number.
        self.decl = line_number(self.source, '// Watchpoint variable declaration.')
        # Build dictionary to have unique executable names for each test method.
        self.exe_name = self.testMethodName
        self.d = {'CXX_SOURCES': self.source, 'EXE': self.exe_name}

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_watchpoint_cond_api_with_dsym(self):
        """Test watchpoint condition API."""
        self.buildDsym(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.watchpoint_condition_api()

    @dwarf_test
    def test_watchpoint_cond_api_with_dwarf(self):
        """Test watchpoint condition API."""
        self.buildDwarf(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.watchpoint_condition_api()

    def watchpoint_condition_api(self):
        """Do watchpoint condition API to set condition as 'global==5'."""
        exe = os.path.join(os.getcwd(), self.exe_name)

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

        # Watch 'global' for write.
        value = frame0.FindValue('global', lldb.eValueTypeVariableGlobal)
        error = lldb.SBError();
        watchpoint = value.Watch(True, False, True, error)
        self.assertTrue(value and watchpoint,
                        "Successfully found the variable and set a watchpoint")
        self.DebugSBValue(value)

        # Now set the condition as "global==5".
        watchpoint.SetCondition('global==5')
        self.expect(watchpoint.GetCondition(), exe=False,
            startstr = 'global==5')

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

        # Verify that the condition is met.
        self.assertTrue(value.GetValueAsUnsigned() == 5)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
