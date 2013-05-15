"""
Use lldb Python SBTarget API to iterate on the watchpoint(s) for the target.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class WatchpointIteratorTestCase(TestBase):

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
    def test_watch_iter_with_dsym(self):
        """Exercise SBTarget.watchpoint_iter() API to iterate on the available watchpoints."""
        self.buildDsym()
        self.do_watchpoint_iter()

    @python_api_test
    @dwarf_test
    def test_watch_iter_with_dwarf(self):
        """Exercise SBTarget.watchpoint_iter() API to iterate on the available watchpoints."""
        self.buildDwarf()
        self.do_watchpoint_iter()

    def do_watchpoint_iter(self):
        """Use SBTarget.watchpoint_iter() to do Pythonic iteration on the available watchpoints."""
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
        self.assertTrue(watchpoint.IsEnabled())
        watch_id = watchpoint.GetID()
        self.assertTrue(watch_id != 0)

        # Continue.  Expect the program to stop due to the variable being written to.
        process.Continue()

        # Hide stdout if not running with '-t' option.
        if not self.TraceOn():
            self.HideStdout()

        # Print the stack traces.
        lldbutil.print_stacktraces(process)

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonWatchpoint)
        self.assertTrue(thread, "The thread stopped due to watchpoint")
        self.DebugSBValue(value)

        # We currently only support hardware watchpoint.  Verify that we have a
        # meaningful hardware index at this point.  Exercise the printed repr of
        # SBWatchpointLocation.
        print watchpoint
        self.assertTrue(watchpoint.GetHardwareIndex() != -1)

        # SBWatchpoint.GetDescription() takes a description level arg.
        print lldbutil.get_description(watchpoint, lldb.eDescriptionLevelFull)

        # Now disable the 'rw' watchpoint.  The program won't stop when it reads
        # 'global' next.
        watchpoint.SetEnabled(False)
        self.assertTrue(watchpoint.GetHardwareIndex() == -1)
        self.assertFalse(watchpoint.IsEnabled())

        # Continue.  The program does not stop again when the variable is being
        # read from because the watchpoint location has been disabled.
        process.Continue()

        # At this point, the inferior process should have exited.
        self.assertTrue(process.GetState() == lldb.eStateExited, PROCESS_EXITED)

        # Verify some vital statistics and exercise the iterator API.
        for watchpoint in target.watchpoint_iter():
            self.assertTrue(watchpoint)
            self.assertTrue(watchpoint.GetWatchSize() == 4)
            self.assertTrue(watchpoint.GetHitCount() == 1)
            print watchpoint


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
