"""
Use lldb Python SBValue API to create a watchpoint for read_write of 'globl' var.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SetWatchpointAPITestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = "main.c"
        # Find the line number to break inside main().
        self.line = line_number(self.source, "// Set break point at this line.")
        self.build()

    # Read-write watchpoints not supported on SystemZ
    @expectedFailureAll(archs=["s390x"])
    def test_watch_val(self):
        """Test watchpoint on a SBValue not backed by a variable yields an expression watchpoint."""
        self._test_watch_val(variable_watchpoint=False)

    @expectedFailureAll(archs=["s390x"])
    def test_watch_variable(self):
        """Test watchpoint on an SBValue backed by a variable yields a variable watchpoint."""
        self._test_watch_val(variable_watchpoint=True)

    @expectedFailureAll(archs=["s390x"])
    def test_local_variable_watchpoint_scoped_to_frame(self):
        """Test watchpoint on a frame local variable only triggers when in the frame scope."""
        local_line = line_number(self.source, "// local_value_breakpoint")
        exe = self.getBuildArtifact("a.out")

        target: lldb.SBTarget = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation(self.source, local_line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)
        self.assertEqual(breakpoint.GetNumLocations(), 1)

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertState(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread, "Stopped at breakpoint inside watch_local()")
        frame = thread.GetSelectedFrame()

        value: lldb.SBValue = frame.FindVariable("local_value")
        self.assertTrue(value.IsValid(), "Found stack-local 'local_value'")

        error = lldb.SBError()
        watchpoint = value.Watch(True, True, True, error)
        self.assertSuccess(error)
        self.assertTrue(watchpoint.IsValid(), "Set watchpoint on 'local_value'")
        self.assertEqual(
            watchpoint.GetWatchValueKind(), lldb.eWatchPointValueKindVariable
        )

        # Continue to the read watchpoint.
        error = process.Continue()
        self.assertSuccess(error)
        self.assertEqual(process.state, lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonWatchpoint)
        self.assertTrue(thread, "stopped at watchpoint read")

        # Verify the process does not stop again after continuing.
        error = process.Continue()
        self.assertTrue(error)
        if process.state != lldb.eStateExited:
            process_state = lldbutil.state_type_to_str(process.state)
            self.fail(f"Process did not run to completion, {process_state=}")

    def _test_watch_val(self, variable_watchpoint):
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
        frame0: lldb.SBFrame = thread.GetFrameAtIndex(0)

        # Watch 'global' for read and write.
        if variable_watchpoint:
            # Variable watchpoint.
            # FindValue returns a variable-backed SBValue,
            value = frame0.FindValue("global", lldb.eValueTypeVariableGlobal)
            error = lldb.SBError()
            watchpoint = value.Watch(True, True, True, error)
            self.assertSuccess(error)
            self.assertTrue(value, VALID_VARIABLE)
            self.assertTrue(watchpoint, f"expected a watchpoint from {value=}")

            self.DebugSBValue(value)
            self.assertEqual(
                watchpoint.GetWatchValueKind(), lldb.eWatchPointValueKindVariable
            )
            self.assertEqual(watchpoint.GetWatchSpec(), value.GetName())
        else:
            # Expression watchpoint.
            # Creates a new SBvalue from an existing SBValue Variable's address and type.
            # This new value its not backed by an actual variable in process memory.
            var_value = frame0.FindValue("global", lldb.eValueTypeVariableGlobal)
            sb_addr = lldb.SBAddress(var_value.GetLoadAddress(), target)
            value = target.CreateValueFromAddress("global", sb_addr, var_value.type)
            error = lldb.SBError()
            watchpoint = value.Watch(True, True, True, error)
            self.assertSuccess(error)
            self.assertTrue(value, VALID_VARIABLE)
            self.assertTrue(watchpoint, f"expected a watchpoint from {value=}")

            self.DebugSBValue(value)
            self.assertEqual(
                watchpoint.GetWatchValueKind(), lldb.eWatchPointValueKindExpression
            )
            # FIXME: The spec should probably be '&global' given that the kind
            # is reported as eWatchPointValueKindExpression. If the kind is
            # actually supposed to be eWatchPointValueKindVariable then the spec
            # should probably be 'global'.
            self.assertEqual(watchpoint.GetWatchSpec(), None)

        self.assertEqual(watchpoint.GetType().GetDisplayTypeName(), "int32_t")
        self.assertEqual(value.GetName(), "global")
        self.assertEqual(value.GetType(), watchpoint.GetType())
        self.assertTrue(watchpoint.IsWatchingReads())
        self.assertTrue(watchpoint.IsWatchingWrites())

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

        # Continue.  Expect the program to stop due to the variable being read
        # from.
        process.Continue()

        if self.TraceOn():
            lldbutil.print_stacktraces(process)

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonWatchpoint)
        self.assertTrue(thread, "The thread stopped due to watchpoint")
        self.DebugSBValue(value)

        # Continue the process.  We don't expect the program to be stopped
        # again.
        process.Continue()

        # At this point, the inferior process should have exited.
        self.assertEqual(process.GetState(), lldb.eStateExited, PROCESS_EXITED)

        self.dbg.DeleteTarget(target)
        self.assertFalse(watchpoint.IsValid())
