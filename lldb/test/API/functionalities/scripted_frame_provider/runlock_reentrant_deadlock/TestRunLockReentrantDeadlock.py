"""
Test that ProcessRunLock re-entrant read locks don't deadlock when a frame
provider's get_frame_at_index calls SB API methods.

This reproduces the deadlock seen in lldb-rpc-server where:

  - An RPC client thread holds a ProcessRunLock read lock (from the outer
    SB API call) and enters a provider's get_frame_at_index, which calls
    SBFrame.IsValid -> GetStoppedExecutionContext -> ReadTryLock (re-entrant).

  - The override PST is exiting RunPrivateStateThread and calls SetStopped
    -> pthread_rwlock_wrlock (blocked by client thread's read lock).

  - The client thread's re-entrant ReadTryLock blocks because the pending
    writer prevents new readers on a write-preferring rwlock.

  - The original PST is blocked joining the override thread.
"""

import os
import threading
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestRunLockReentrantDeadlock(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24528 (scripted breakpoint resolvers fail on "
        "Windows; same XFAIL as the sibling tests in this directory)",
    )
    def test_runlock_reentrant_no_deadlock(self):
        """
        Test that a frame provider calling SBFrame.IsValid from
        get_frame_at_index does not deadlock with the override PST's
        SetStopped when another thread triggers EvaluateExpression.
        """
        self.build()

        target, process, thread, _ = lldbutil.run_to_name_breakpoint(self, "main")

        resolver_path = os.path.join(self.getSourceDir(), "bkpt_resolver.py")
        provider_path = os.path.join(self.getSourceDir(), "frame_provider.py")
        self.runCmd("command script import " + resolver_path)
        self.runCmd("command script import " + provider_path)

        error = lldb.SBError()
        provider_id = target.RegisterScriptedFrameProvider(
            "frame_provider.SBAPIAccessInGetFrameProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(
            error.Success(),
            f"Should register frame provider: {error}",
        )
        self.assertNotEqual(provider_id, 0, "Provider ID should be non-zero")

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

        # Spawn a thread that repeatedly accesses frames through the provider.
        # This simulates an RPC client thread that holds the ProcessRunLock
        # read lock and enters get_frame_at_index -> SBFrame.IsValid.
        stop_frame_access = threading.Event()
        frame_access_error = [None]

        def access_frames():
            try:
                while not stop_frame_access.is_set():
                    t = process.GetSelectedThread()
                    if t.IsValid():
                        for i in range(t.GetNumFrames()):
                            f = t.GetFrameAtIndex(i)
                            f.IsValid()
            except Exception as e:
                frame_access_error[0] = str(e)

        frame_thread = threading.Thread(target=access_frames, daemon=True)
        frame_thread.start()

        # Continue into the breakpoint. was_hit calls EvaluateExpression on
        # the PST, which triggers RunThreadPlan -> override PST.
        # The frame access thread is concurrently calling GetFrameAtIndex ->
        # provider get_frame_at_index -> SBFrame.IsValid.
        # Without the fix, this deadlocks.
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)

        stop_frame_access.set()
        frame_thread.join(timeout=5)
        self.assertFalse(frame_thread.is_alive(), "Frame access thread should exit")
        self.assertIsNone(
            frame_access_error[0],
            f"Frame access thread hit an error: {frame_access_error[0]}",
        )

        thread = process.GetSelectedThread()
        self.assertTrue(thread.IsValid(), "Thread should be valid")
        self.assertEqual(
            thread.GetStopReason(),
            lldb.eStopReasonBreakpoint,
            "Should stop at breakpoint",
        )

        g_value = target.FindFirstGlobalVariable("g_value")
        self.assertTrue(g_value.IsValid(), "Should find g_value")
        self.assertGreater(
            g_value.GetValueAsUnsigned(),
            0,
            "g_value should have been incremented by the was_hit callback",
        )
