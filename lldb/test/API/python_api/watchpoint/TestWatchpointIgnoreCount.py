"""
Use lldb Python SBWatchpoint API to set the ignore count.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class WatchpointIgnoreCountTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.c'
        # Find the line number to break inside main().
        self.line = line_number(
            self.source, '// Set break point at this line.')

    # on arm64 targets, lldb has incorrect hit-count / ignore-counts
    # for watchpoints when they are hit with multiple threads at
    # the same time.  Tracked as llvm.org/pr49433
    # or rdar://93863107 inside Apple.
    def affected_by_radar_93863107(self):
        return (self.getArchitecture() in ['arm64', 'arm64e']) and self.platformIsDarwin()

    # Read-write watchpoints not supported on SystemZ
    @expectedFailureAll(archs=['s390x'])
    def test_set_watch_ignore_count(self):
        """Test SBWatchpoint.SetIgnoreCount() API."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create a breakpoint on main.c in order to set our watchpoint later.
        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        # We should be stopped due to the breakpoint.  Get frame #0.
        process = target.GetProcess()
        self.assertEqual(process.GetState(), lldb.eStateStopped,
                        PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        frame0 = thread.GetFrameAtIndex(0)

        # Watch 'global' for read and write.
        value = frame0.FindValue('global', lldb.eValueTypeVariableGlobal)
        error = lldb.SBError()
        watchpoint = value.Watch(True, True, True, error)
        self.assertTrue(value and watchpoint,
                        "Successfully found the variable and set a watchpoint")
        self.DebugSBValue(value)

        # Hide stdout if not running with '-t' option.
        if not self.TraceOn():
            self.HideStdout()

        # There should be only 1 watchpoint location under the target.
        self.assertEqual(target.GetNumWatchpoints(), 1)
        watchpoint = target.GetWatchpointAtIndex(0)
        self.assertTrue(watchpoint.IsEnabled())
        self.assertEqual(watchpoint.GetIgnoreCount(), 0)
        watch_id = watchpoint.GetID()
        self.assertNotEqual(watch_id, 0)
        print(watchpoint)

        # Now immediately set the ignore count to 2.  When we continue, expect the
        # inferior to run to its completion without stopping due to watchpoint.
        watchpoint.SetIgnoreCount(2)
        print(watchpoint)
        process.Continue()

        # At this point, the inferior process should have exited.
        self.assertEqual(process.GetState(), lldb.eStateExited, PROCESS_EXITED)

        # Verify some vital statistics.
        self.assertTrue(watchpoint)
        self.assertEqual(watchpoint.GetWatchSize(), 4)
        if not self.affected_by_radar_93863107():
          self.assertEqual(watchpoint.GetHitCount(), 2)
        print(watchpoint)
