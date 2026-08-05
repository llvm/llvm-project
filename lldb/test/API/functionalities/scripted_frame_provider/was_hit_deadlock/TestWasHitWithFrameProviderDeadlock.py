"""
Test that a scripted breakpoint's was_hit callback calling EvaluateExpression
does not deadlock when a scripted frame provider is also registered.

This reproduces the deadlock from:
    lldb-rpc-server sample showing two private state threads in mutual wait:

    Thread A (lldb.process.internal-state):
      BreakpointResolverScripted::WasHit -> Python -> SBFrame.EvaluateExpression
        -> RunThreadPlan -> Halt -> WaitForProcessToStop
        (holds mutex, waits for state change event)

    Thread B (lldb.process.internal-state-override):
      HandlePrivateEvent -> ShouldStop -> GetStackFrameList
        -> LoadScriptedFrameProvider -> ScriptedFrameProvider.__init__
        -> Python -> SBThread.__bool__ -> GetStoppedExecutionContext
        -> recursive_mutex::lock (BLOCKED - mutex held by Thread A)

    Thread A waits for the event that Thread B would deliver, but Thread B
    is blocked on the mutex Thread A holds.
"""

import os
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestWasHitWithFrameProviderDeadlock(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24528")
    def test_was_hit_with_frame_provider_no_deadlock(self):
        """
        Test that a scripted breakpoint doing EvaluateExpression in was_hit
        does not deadlock when a scripted frame provider is registered.
        """
        self.build()

        target, process, thread, _ = lldbutil.run_to_name_breakpoint(self, "main")

        # Import both the breakpoint resolver and the frame provider scripts.
        resolver_path = os.path.join(self.getSourceDir(), "bkpt_resolver.py")
        provider_path = os.path.join(self.getSourceDir(), "frame_provider.py")
        self.runCmd("command script import " + resolver_path)
        self.runCmd("command script import " + provider_path)

        # Register the scripted frame provider FIRST.
        # This means that when any stop event is processed and the thread's
        # stack frames are accessed, the provider will be loaded, calling
        # its __init__ which accesses SBThread.
        error = lldb.SBError()
        provider_id = target.RegisterScriptedFrameProvider(
            "frame_provider.SBAPIAccessInInitProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(
            error.Success(),
            f"Should register frame provider: {error}",
        )
        self.assertNotEqual(provider_id, 0, "Provider ID should be non-zero")

        # Create the scripted breakpoint that calls EvaluateExpression in was_hit.
        extra_args = lldb.SBStructuredData()
        stream = lldb.SBStream()
        stream.Print('{"symbol": "target_func"}')
        extra_args.SetFromJSON(stream)

        bkpt = target.BreakpointCreateFromScript(
            "bkpt_resolver.ExprEvalResolver",
            extra_args,
            lldb.SBFileSpecList(),
            lldb.SBFileSpecList(),
        )
        self.assertTrue(bkpt.IsValid(), "Scripted breakpoint should be valid")
        self.assertGreater(
            bkpt.GetNumLocations(), 0, "Breakpoint should have locations"
        )

        # Continue the process. When the breakpoint is hit:
        # 1. was_hit runs on private state thread, calls EvaluateExpression
        # 2. EvaluateExpression -> RunThreadPlan -> Halt -> WaitForProcessToStop
        #    (holds mutex, waits for state event)
        # 3. Override state thread processes the stop, loads frame provider
        # 4. Provider __init__ calls SBThread.__bool__ -> GetStoppedExecutionContext
        #    -> tries to acquire mutex held by step 2 -> DEADLOCK
        #
        # If this test completes without hanging, the deadlock is fixed.
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)

        thread = process.GetSelectedThread()
        self.assertTrue(thread.IsValid(), "Thread should be valid")
        self.assertEqual(
            thread.GetStopReason(),
            lldb.eStopReasonBreakpoint,
            "Should stop at breakpoint",
        )

        # Verify that EvaluateExpression in was_hit actually ran successfully
        # by checking that g_value was incremented.
        g_value = target.FindFirstGlobalVariable("g_value")
        self.assertTrue(g_value.IsValid(), "Should find g_value")
        self.assertGreater(
            g_value.GetValueAsUnsigned(),
            0,
            "g_value should have been incremented by the was_hit callback",
        )

        # Continue to hit the breakpoint again to ensure it's not a one-off.
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertEqual(
            thread.GetStopReason(),
            lldb.eStopReasonBreakpoint,
            "Should stop at breakpoint again",
        )
