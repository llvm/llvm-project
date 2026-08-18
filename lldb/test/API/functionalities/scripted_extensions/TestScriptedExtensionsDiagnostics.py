"""
Verify that exceptions raised inside scripted extension affordance methods
(or missing required abstract methods) are surfaced to the user.

For entry points with no return-channel for errors
(`ScriptedProcess::CreateInstance`, `OperatingSystemPython` ctor,
`ScriptedThreadPlan::DidPush`, `BreakpointResolverScripted` ctor,
`ScriptedStackFrameRecognizer` ctor) the diagnostic is broadcast via
`Debugger::ReportError` and asserted on a listener.

`ScriptedThread::Create` and `ScriptedFrame::Create` also return an
`llvm::Expected`, but their only callers
(`ScriptedProcess::DoUpdateThreadList` and
`ScriptedThread::LoadArtificialStackFrames`, respectively) have no
return-channel of their own, so those errors are likewise broadcast via
`Debugger::ReportError` rather than propagated further.

Entry points that already return an `llvm::Expected` / `Status` all the way
to a user-visible surface (`ScriptedFrameProvider::CreateInstance`,
`StopHookScripted::SetScriptCallback`) propagate the detailed error through
their return type; tests for those are tracked as follow-up.
"""

import os

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import expectedFailureAll
from lldbsuite.test.lldbtest import TestBase


class TestScriptedExtensionsDiagnostics(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.broadcaster = self.dbg.GetBroadcaster()
        self.listener = lldbutil.start_listening_from(
            self.broadcaster,
            lldb.SBDebugger.eBroadcastBitWarning | lldb.SBDebugger.eBroadcastBitError,
        )
        script_path = os.path.join(
            self.getSourceDir(), "malformed_scripted_extensions.py"
        )
        self.runCmd("command script import " + script_path)

    def assert_diagnostic(self, expected_substring):
        event = lldbutil.fetch_next_event(self, self.listener, self.broadcaster)
        data = lldb.SBDebugger.GetDiagnosticFromEvent(event)
        self.assertTrue(data.IsValid(), "event has diagnostic data")
        message = data.GetValueForKey("message").GetStringValue(4096)
        self.assertIn(expected_substring, message)

    def create_target(self):
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, "valid target")
        return target

    # ------------------------------------------------------------------
    # ScriptedProcess - reports via ScriptedProcess::CreateInstance
    # ------------------------------------------------------------------

    def test_scripted_process_missing_methods(self):
        """A ScriptedProcess missing abstract methods should emit a
        diagnostic naming the missing method."""
        target = self.create_target()
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetProcessPluginName("ScriptedProcess")
        launch_info.SetScriptedProcessClassName(
            "malformed_scripted_extensions.MissingMethodsScriptedProcess"
        )
        error = lldb.SBError()
        target.Launch(launch_info, error)
        self.assertTrue(error.Fail(), "launch should fail")
        self.assert_diagnostic("read_memory_at_address")

    # ------------------------------------------------------------------
    # ScriptedProcess::launch() - reports via the normal SBError return
    # channel (Process::DoLaunch), unlike CreateInstance's construction
    # failures above, which have no return channel and go through
    # Debugger::ReportError instead.
    # ------------------------------------------------------------------

    def test_scripted_process_launch_exception(self):
        """An explicitly raised exception from `launch()` should surface
        through the normal SBError return channel."""
        target = self.create_target()
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetProcessPluginName("ScriptedProcess")
        launch_info.SetScriptedProcessClassName(
            "malformed_scripted_extensions.ExceptionScriptedProcess"
        )
        error = lldb.SBError()
        target.Launch(launch_info, error)
        self.assertTrue(error.Fail(), "launch should fail")
        self.assertIn("intentional exception from launch()", error.GetCString())

    def test_scripted_process_launch_runtime_error(self):
        """A natural Python runtime error (e.g. a typo'd name), as opposed
        to a deliberately raised exception, should surface just as
        readably: this is the other half of what users hit in practice --
        either an implementation function returns something that doesn't
        make sense, or (this case) they mistyped a name and Python
        complains about it at runtime."""
        target = self.create_target()
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetProcessPluginName("ScriptedProcess")
        launch_info.SetScriptedProcessClassName(
            "malformed_scripted_extensions.TypoScriptedProcess"
        )
        error = lldb.SBError()
        target.Launch(launch_info, error)
        self.assertTrue(error.Fail(), "launch should fail")
        self.assertIn("NameError", error.GetCString())
        self.assertIn("this_name_is_never_defined", error.GetCString())

    # ------------------------------------------------------------------
    # BreakpointResolverScripted - reports via
    # BreakpointResolverScripted::CreateImplementationIfNeeded.
    # `m_error` is set but never surfaced to the user, so ReportError is
    # the only user-visible channel.
    # ------------------------------------------------------------------

    def test_scripted_breakpoint_resolver_init_failure(self):
        target = self.create_target()
        target.BreakpointCreateFromScript(
            "malformed_scripted_extensions.ExceptionInitScriptedBreakpointResolver",
            lldb.SBStructuredData(),
            lldb.SBFileSpecList(),
            lldb.SBFileSpecList(),
        )
        self.assert_diagnostic("intentional exception from __init__()")

    # ------------------------------------------------------------------
    # ScriptedThreadPlan - reports via ScriptedThreadPlan::DidPush. The
    # plan stores the error in m_error_str but never surfaces it; the
    # diagnostic is the user-visible channel.
    # ------------------------------------------------------------------

    def test_scripted_thread_plan_init_failure(self):
        target = self.create_target()
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, "valid process")
        thread = process.GetSelectedThread()
        self.assertTrue(thread, "valid thread")
        thread.StepUsingScriptedThreadPlan(
            "malformed_scripted_extensions.ExceptionInitScriptedThreadPlan"
        )
        self.assert_diagnostic("intentional exception from __init__()")

    # ------------------------------------------------------------------
    # ScriptedThread - reports via ScriptedProcess::DoUpdateThreadList,
    # which propagates ScriptedThread::Create's Expected error through
    # Debugger::ReportError (DoUpdateThreadList has no return-channel back
    # to its caller).
    # ------------------------------------------------------------------

    def test_scripted_thread_missing_methods(self):
        """A scripted thread object returned from `get_threads_info()` that
        is missing a required abstract method should emit a diagnostic
        naming the missing method."""
        target = self.create_target()
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetProcessPluginName("ScriptedProcess")
        launch_info.SetScriptedProcessClassName(
            "malformed_scripted_extensions.ThreadListScriptedProcess"
        )
        error = lldb.SBError()
        target.Launch(launch_info, error)
        self.assert_diagnostic("get_stop_reason")

    # ------------------------------------------------------------------
    # ScriptedFrame - reports via ScriptedThread::LoadArtificialStackFrames,
    # which propagates ScriptedFrame::Create's Expected error through
    # Debugger::ReportError (LoadArtificialStackFrames' return value is
    # discarded by its only caller, RefreshStateAfterStop).
    # ------------------------------------------------------------------

    def test_scripted_frame_missing_methods(self):
        """A scripted frame object returned from a thread's
        `get_stackframes()` that is missing a required abstract method
        should emit a diagnostic naming the missing method."""
        target = self.create_target()
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetProcessPluginName("ScriptedProcess")
        launch_info.SetScriptedProcessClassName(
            "malformed_scripted_extensions.StackFrameScriptedProcess"
        )
        error = lldb.SBError()
        target.Launch(launch_info, error)
        self.assert_diagnostic("get_id")

    # ------------------------------------------------------------------
    # ScriptedStackFrameRecognizer - reports via
    # ScriptedStackFrameRecognizer's constructor, which has no
    # error-return channel back to `frame recognizer add`.
    # ------------------------------------------------------------------

    def test_scripted_stack_frame_recognizer_init_failure(self):
        target = self.create_target()
        self.runCmd(
            "frame recognizer add -l "
            "malformed_scripted_extensions.ExceptionScriptedStackFrameRecognizer "
            "-s a.out -n main"
        )
        self.assert_diagnostic("intentional exception from __init__()")

    # ------------------------------------------------------------------
    # The remaining entry point has no plugin implementation yet.
    # ------------------------------------------------------------------

    def test_operating_system_missing_methods(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))
        os_plugin_path = os.path.join(
            self.getSourceDir(), "os_plugin_missing_methods.py"
        )
        self.runCmd(
            "settings set target.process.python-os-plugin-path " + os_plugin_path
        )
        self.runCmd("thread list")
        self.assert_diagnostic("get_thread_info")

    @expectedFailureAll(bugnumber="ScriptedPlatform has no plugin implementation yet")
    def test_scripted_platform_missing_methods(self):
        self.assert_diagnostic("list_processes")
