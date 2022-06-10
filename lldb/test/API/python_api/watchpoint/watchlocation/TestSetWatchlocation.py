"""
Use lldb Python SBValue.WatchPointee() API to create a watchpoint for write of '*g_char_ptr'.
"""

from __future__ import print_function



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SetWatchlocationAPITestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.cpp'
        # Find the line number to break inside main().
        self.line = line_number(
            self.source, '// Set break point at this line.')
        # This is for verifying that watch location works.
        self.violating_func = "do_bad_thing_with_location"

    @skipIfWindows # This test is flaky on Windows
    def test_watch_location(self):
        """Exercise SBValue.WatchPointee() API to set a watchpoint."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c.
        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        # We should be stopped due to the breakpoint.  Get frame #0.
        process = target.GetProcess()
        self.assertState(process.GetState(), lldb.eStateStopped,
                         PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        frame0 = thread.GetFrameAtIndex(0)

        value = frame0.FindValue('g_char_ptr',
                                 lldb.eValueTypeVariableGlobal)
        pointee = value.CreateValueFromAddress(
            "pointee",
            value.GetValueAsUnsigned(0),
            value.GetType().GetPointeeType())
        # Watch for write to *g_char_ptr.
        error = lldb.SBError()
        watchpoint = value.WatchPointee(True, False, True, error)
        self.assertTrue(value and watchpoint,
                        "Successfully found the pointer and set a watchpoint")
        self.DebugSBValue(value)
        self.DebugSBValue(pointee)

        # Hide stdout if not running with '-t' option.
        if not self.TraceOn():
            self.HideStdout()

        print(watchpoint)

        # Continue.  Expect the program to stop due to the variable being
        # written to.
        process.Continue()

        if (self.TraceOn()):
            lldbutil.print_stacktraces(process)

        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonWatchpoint)
        self.assertTrue(thread, "The thread stopped due to watchpoint")
        self.DebugSBValue(value)
        self.DebugSBValue(pointee)

        self.expect(
            lldbutil.print_stacktrace(
                thread,
                string_buffer=True),
            exe=False,
            substrs=[
                self.violating_func])

        # This finishes our test.
