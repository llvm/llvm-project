"""
Test that an expression fails cleanly when its thread exits while it runs, and
that the process can continue afterwards.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


@requireThreadSupport
@requireNotWasm("main.c continues past __builtin_debugtrap, a fatal trap on Wasm.")
@skipIfAsan
@skipIfWindows
class ExitDuringExpressionTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(oslist=["linux"], archs=["arm$", "aarch64"], bugnumber="llvm.org/pr48414")
    @expectedFailureAll(oslist=["freebsd"], bugnumber="llvm.org/pr48414")
    @expectedFailureNetBSD
    def test_exit_before_one_thread_timeout_unwind(self):
        self.do_test(exit_before_one_thread_timeout=True, unwind=True)

    @skipIf(oslist=["linux"], archs=["arm$", "aarch64"], bugnumber="llvm.org/pr48414")
    @expectedFailureAll(oslist=["freebsd"], bugnumber="llvm.org/pr48414")
    @expectedFailureNetBSD
    def test_exit_before_one_thread_timeout_no_unwind(self):
        self.do_test(exit_before_one_thread_timeout=True, unwind=False)

    def test_exit_after_one_thread_timeout_unwind(self):
        self.do_test(exit_before_one_thread_timeout=False, unwind=True)

    def test_exit_after_one_thread_timeout_no_unwind(self):
        self.do_test(exit_before_one_thread_timeout=False, unwind=False)

    def do_test(self, exit_before_one_thread_timeout, unwind):
        target, process, thread, _ = self.build_and_run()
        tid = thread.GetThreadID()

        # Keep the thread alive until lldb resumes all threads.
        if not exit_before_one_thread_timeout:
            wait_for_release = target.FindFirstGlobalVariable("g_wait_for_release")
            error = lldb.SBError()
            wait_for_release.SetValueFromCString("1", error)
            self.assertSuccess(error)

        options = lldb.SBExpressionOptions()
        options.SetUnwindOnError(unwind)
        # The thread exits immediately in the first case, and the second case
        # only depends on the one-thread timeout having passed.
        options.SetOneThreadTimeoutInMicroSeconds(100 * 1000)
        # Only an upper bound; the expression ends as soon as the thread exits.
        options.SetTimeoutInMicroSeconds(60 * 1000 * 1000)

        result = self.frame().EvaluateExpression("exit_current_thread()", options)
        # The thread-exit error, not a timeout.
        self.assertIn("exited during its execution", str(result.GetError()))
        self.assertFalse(process.GetThreadByID(tid).IsValid(), "the thread exited")

        if exit_before_one_thread_timeout:
            # main is still waiting for the thread in pthread_join.
            process.Continue()
        # Traps are reported as exceptions on Darwin and signals elsewhere.
        trap_stop_reason = lldb.eStopReasonSignal
        if self.platformIsDarwin():
            trap_stop_reason = lldb.eStopReasonException
        main_thread = lldbutil.get_stopped_thread(process, trap_stop_reason)
        self.assertIsNotNone(main_thread, "main stopped at the trap")
        thread_exited = main_thread.GetFrameAtIndex(0).FindVariable("thread_exited")
        self.assertEqual(thread_exited.GetValueAsUnsigned(), 1)

        # main returns 0 only if the thread exited instead of returning.
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)
