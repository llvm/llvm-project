"""
Use lldb Python SBWatchpoint API to set the ignore count.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class WatchpointIgnoreCountTestCase(TestBase):

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
    def test_set_watch_ignore_count_with_dsym(self):
        """Test SBWatchpoint.SetIgnoreCount() API."""
        self.buildDsym()
        self.do_watchpoint_ignore_count()

    @python_api_test
    @dwarf_test
    def test_set_watch_ignore_count_with_dwarf(self):
        """Test SBWatchpoint.SetIgnoreCount() API."""
        self.buildDwarf()
        self.do_watchpoint_ignore_count()

    def do_watchpoint_ignore_count(self):
        """Test SBWatchpoint.SetIgnoreCount() API."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create a breakpoint on main.c in order to set our watchpoint later.
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

        # There should be only 1 watchpoint location under the target.
        self.assertTrue(target.GetNumWatchpoints() == 1)
        watchpoint = target.GetWatchpointAtIndex(0)
        self.assertTrue(watchpoint.IsEnabled())
        self.assertTrue(watchpoint.GetIgnoreCount() == 0)
        watch_id = watchpoint.GetID()
        self.assertTrue(watch_id != 0)
        print watchpoint

        # Now immediately set the ignore count to 2.  When we continue, expect the
        # inferior to run to its completion without stopping due to watchpoint.
        watchpoint.SetIgnoreCount(2)
        print watchpoint
        process.Continue()

        # At this point, the inferior process should have exited.
        self.assertTrue(process.GetState() == lldb.eStateExited, PROCESS_EXITED)

        # Verify some vital statistics.
        self.assertTrue(watchpoint)
        self.assertTrue(watchpoint.GetWatchSize() == 4)
        self.assertTrue(watchpoint.GetHitCount() == 2)
        print watchpoint


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
