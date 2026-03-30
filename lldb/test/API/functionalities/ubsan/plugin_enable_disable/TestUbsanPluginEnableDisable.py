"""
Tests enabling and disabling the UndefinedBehaviorSanitizer instrumentation
runtime plugin during a debug session.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class UbsanPluginEnableDisableTestCase(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.line_first_ubsan_issue = line_number("main.c", "// first ubsan issue")
        self.line_third_ubsan_issue = line_number("main.c", "// third ubsan issue")

    def ubsan_plugin_is_enabled(self):
        return self.plugin_is_enabled(
            "instrumentation-runtime.UndefinedBehaviorSanitizer"
        )

    def check_stopped_at_ubsan_issue(self, line_num):
        process = self.dbg.GetSelectedTarget().process
        thread = process.GetSelectedThread()
        stop_reason = thread.GetStopReason()
        self.assertStopReason(stop_reason, lldb.eStopReasonInstrumentation)
        self.assertIn("__ubsan_on_report", thread.GetFrameAtIndex(0).GetFunctionName())
        backtraces = thread.GetStopReasonExtendedBacktraces(
            lldb.eInstrumentationRuntimeTypeUndefinedBehaviorSanitizer
        )
        self.assertEqual(backtraces.GetSize(), 1)

        # Check that we stopped at the expected line somewhere in the stacktrace
        found = False
        for i in range(thread.GetNumFrames()):
            frame = thread.GetFrameAtIndex(i)
            if frame.GetLineEntry().GetFileSpec().GetFilename() == "main.c":
                if frame.GetLineEntry().GetLine() == line_num:
                    found = True
        self.assertTrue(found)

    @skipUnlessUndefinedBehaviorSanitizer
    @no_debug_info_test
    def test_disable_plugin_after_hit(self):
        """Test that disabling the UBSan plugin mid-session prevents further
        instrumentation stops."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.registerSanitizerLibrariesWithTarget(target)

        self.runCmd("run")
        self.assertTrue(self.ubsan_plugin_is_enabled())

        process = self.dbg.GetSelectedTarget().process
        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()

        # We should have stopped on the first UBSan issue.
        self.check_stopped_at_ubsan_issue(self.line_first_ubsan_issue)

        # Disable the UBSan plugin.
        self.runCmd("plugin disable instrumentation-runtime.UndefinedBehaviorSanitizer")
        self.assertFalse(self.ubsan_plugin_is_enabled())

        # Continue. The remaining UBSan issues should not cause
        # instrumentation stops and the process should exit cleanly.
        try:
            process.Continue()
            self.assertEqual(process.GetState(), lldb.eStateExited)
            self.assertEqual(process.GetExitStatus(), 0)
        finally:
            # Restore the default so we don't affect other tests. This command
            # affects the debugger session globally so we have to be careful
            # to restore the global state after we are done.
            self.runCmd(
                "plugin enable instrumentation-runtime.UndefinedBehaviorSanitizer"
            )
            self.assertTrue(self.ubsan_plugin_is_enabled())

    @skipUnlessUndefinedBehaviorSanitizer
    @no_debug_info_test
    def test_enable_plugin_after_disable(self):
        """Test that disabling the UBSan plugin before launch prevents
        instrumentation stops, and re-enabling it mid-session restores them."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.registerSanitizerLibrariesWithTarget(target)

        # Disable the UBSan plugin before launching the process.
        self.runCmd("plugin disable instrumentation-runtime.UndefinedBehaviorSanitizer")
        self.assertFalse(self.ubsan_plugin_is_enabled())

        # Set a breakpoint on test_breakpoint which is called just before
        # the last UBSan issue.
        bp = target.BreakpointCreateByName("test_breakpoint")
        self.assertTrue(bp.GetNumLocations() > 0)

        self.runCmd("run")

        process = self.dbg.GetSelectedTarget().process
        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()

        # We should have skipped most UBSan issues and stopped at the
        # test_breakpoint function.
        stop_reason = thread.GetStopReason()
        self.assertStopReason(stop_reason, lldb.eStopReasonBreakpoint)
        self.assertIn("test_breakpoint", frame.GetFunctionName())

        # Re-enable the UBSan plugin.
        self.runCmd("plugin enable instrumentation-runtime.UndefinedBehaviorSanitizer")
        self.assertTrue(self.ubsan_plugin_is_enabled())

        # Continue
        process.Continue()
        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()

        # We should now hit the last UBSan issue.
        self.check_stopped_at_ubsan_issue(self.line_third_ubsan_issue)

        # Continue. The process should exit cleanly.
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)
        self.assertTrue(self.ubsan_plugin_is_enabled())
