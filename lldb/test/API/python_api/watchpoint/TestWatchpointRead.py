"""
Use lldb Python SBTarget API to set read watchpoints
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SetReadOnlyWatchpointTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = "main.c"
        # Find the line number to break inside main().
        self.line = line_number(self.source, "// Set break point at this line.")
        self.build()

    # Intel hardware does not support read-only watchpoints
    @expectedFailureAll(archs=["i386", "x86_64"])
    def test_read_watchpoint_watch_address(self):
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c.
        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(
            breakpoint and breakpoint.GetNumLocations() == 1, VALID_BREAKPOINT
        )

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())

        # We should be stopped due to the breakpoint.  Get frame #0.
        process = target.GetProcess()
        self.assertState(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        frame0 = thread.GetFrameAtIndex(0)

        value = frame0.FindValue("global", lldb.eValueTypeVariableGlobal)
        local = frame0.FindValue("local", lldb.eValueTypeVariableLocal)
        error = lldb.SBError()

        watchpoint = target.WatchAddress(value.GetLoadAddress(), 1, True, False, error)
        self.assertTrue(
            value and local and watchpoint,
            "Successfully found the values and set a watchpoint",
        )
        self.DebugSBValue(value)
        self.DebugSBValue(local)

        # Hide stdout if not running with '-t' option.
        if not self.TraceOn():
            self.HideStdout()

        print(watchpoint)

        # Continue.  Expect the program to stop due to the variable being
        # read, but *not* written to.
        process.Continue()

        if self.TraceOn():
            lldbutil.print_stacktraces(process)

        self.assertTrue(
            local.GetValueAsSigned() > 0, "The local variable has been incremented"
        )

    # Intel hardware does not support read-only watchpoints
    @expectedFailureAll(archs=["i386", "x86_64"])
    def test_read_watchpoint_watch_create_by_address(self):
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c.
        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(
            breakpoint and breakpoint.GetNumLocations() == 1, VALID_BREAKPOINT
        )

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())

        # We should be stopped due to the breakpoint.  Get frame #0.
        process = target.GetProcess()
        self.assertState(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        frame0 = thread.GetFrameAtIndex(0)

        value = frame0.FindValue("global", lldb.eValueTypeVariableGlobal)
        local = frame0.FindValue("local", lldb.eValueTypeVariableLocal)
        error = lldb.SBError()

        wp_opts = lldb.SBWatchpointOptions()
        wp_opts.SetWatchpointTypeRead(True)
        watchpoint = target.WatchpointCreateByAddress(
            value.GetLoadAddress(), 1, wp_opts, error
        )
        self.assertTrue(
            value and local and watchpoint,
            "Successfully found the values and set a watchpoint",
        )
        self.DebugSBValue(value)
        self.DebugSBValue(local)

        # Hide stdout if not running with '-t' option.
        if not self.TraceOn():
            self.HideStdout()

        print(watchpoint)

        # Continue.  Expect the program to stop due to the variable being
        # read, but *not* written to.
        process.Continue()

        if self.TraceOn():
            lldbutil.print_stacktraces(process)

        self.assertTrue(
            local.GetValueAsSigned() > 0, "The local variable has been incremented"
        )
