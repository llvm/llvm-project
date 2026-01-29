"""
Use lldb Python SBtarget.WatchpointCreateByAddress() API to create a watchpoint for write of '*g_char_ptr'.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbwatchpointutils import *


class TargetWatchpointCreateByAddressPITestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = "main.cpp"
        # Find the line number to break inside main().
        self.line = line_number(self.source, "// Set break point at this line.")
        # This is for verifying that watch location works.
        self.violating_func = "do_bad_thing_with_location"

    @skipIf(
        oslist=["windows"],
        archs=["x86_64"],
        bugnumber="github.com/llvm/llvm-project/issues/144777",
    )
    @expectedFailureAll(archs="^riscv.*")
    def test_hw_watch_address(self):
        self.do_watch_address(WatchpointType.MODIFY, lldb.eWatchpointModeHardware)

    @skip
    def test_sw_watch_address(self):
        self.do_watch_address(WatchpointType.MODIFY, lldb.eWatchpointModeSoftware)

    def do_watch_address(self, wp_type, wp_mode):
        """Exercise SBTarget.WatchpointCreateByAddress() API to set a watchpoint."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
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

        value = frame0.FindValue("g_char_ptr", lldb.eValueTypeVariableGlobal)
        pointee = value.CreateValueFromAddress(
            "pointee", value.GetValueAsUnsigned(0), value.GetType().GetPointeeType()
        )
        # Watch for write to *g_char_ptr.
        error = lldb.SBError()
        watchpoint = set_watchpoint_by_address(
            target, value.GetValueAsUnsigned(), 1, wp_type, wp_mode, error
        )
        self.assertTrue(
            value and watchpoint, "Successfully found the pointer and set a watchpoint"
        )
        self.DebugSBValue(value)
        self.DebugSBValue(pointee)

        # Hide stdout if not running with '-t' option.
        if not self.TraceOn():
            self.HideStdout()

        print(watchpoint)

        # Continue.  Expect the program to stop due to the variable being
        # written to.
        process.Continue()

        if self.TraceOn():
            lldbutil.print_stacktraces(process)

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonWatchpoint)
        self.assertTrue(thread, "The thread stopped due to watchpoint")
        self.DebugSBValue(value)
        self.DebugSBValue(pointee)

        self.expect(
            lldbutil.print_stacktrace(thread, string_buffer=True),
            exe=False,
            substrs=[self.violating_func],
        )

        # This finishes our test.

    # No size constraint on MIPS for watches
    @skipIf(archs=["mips", "mipsel", "mips64", "mips64el"])
    @skipIf(archs=["s390x"])  # Likewise on SystemZ
    @skipIf(
        oslist=["windows"],
        archs=["x86_64"],
        bugnumber="github.com/llvm/llvm-project/issues/142196",
    )
    @expectedFailureAll(archs="^riscv.*")
    def test_hw_watch_address_with_invalid_watch_size(self):
        self.do_watch_address_with_invalid_watch_size(
            WatchpointType.MODIFY, lldb.eWatchpointModeHardware
        )

    def test_sw_watch_address_with_invalid_watch_size(self):
        self.do_watch_address_with_invalid_watch_size(
            WatchpointType.MODIFY, lldb.eWatchpointModeSoftware
        )

    def do_watch_address_with_invalid_watch_size(self, wp_type, wp_mode):
        """Exercise SBTarget.WatchpointCreateByAddress() API but pass an invalid watch_size."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
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

        value = frame0.FindValue("g_char_ptr", lldb.eValueTypeVariableGlobal)
        pointee = value.CreateValueFromAddress(
            "pointee", value.GetValueAsUnsigned(0), value.GetType().GetPointeeType()
        )

        # debugserver on Darwin AArch64 systems can watch large regions
        # of memory via https://reviews.llvm.org/D149792 , don't run this
        # test there.
        if wp_mode == lldb.eWatchpointModeHardware and (
            self.getArchitecture() not in ["arm64", "arm64e", "arm64_32"]
        ):
            # Watch for write to *g_char_ptr.
            error = lldb.SBError()
            watchpoint = set_watchpoint_by_address(
                target, value.GetValueAsUnsigned(), 365, wp_type, wp_mode, error
            )
            self.assertFalse(watchpoint)
            self.expect(
                error.GetCString(),
                exe=False,
                substrs=["Setting one of the watchpoint resources failed"],
            )
