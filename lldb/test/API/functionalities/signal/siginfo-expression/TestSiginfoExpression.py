"""Test that evaluating expressions does not change the siginfo of a thread."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipUnlessPlatform(["linux"])
class SiginfoExpressionTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def launch(self):
        self.build()
        target = self.createTestTarget()
        lldbutil.run_break_set_by_source_regexp(self, "break in handler")
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonSignal)
        self.assertTrue(thread.IsValid())
        return thread

    def check_siginfo(self, thread):
        """Check that the thread still describes the original crash, and that
        the signal handler gets the same information."""
        self.assertIn("fault address=0x1234", thread.GetStopDescription(1000))

        process = thread.GetProcess()
        siginfo = thread.GetSiginfo()
        self.assertSuccess(siginfo.GetError())
        self.assertEqual(
            siginfo.GetChildMemberWithName("si_signo").GetValueAsUnsigned(),
            process.GetUnixSignals().GetSignalNumberFromName("SIGSEGV"),
        )
        sigfault = siginfo.GetChildMemberWithName("_sifields").GetChildMemberWithName(
            "_sigfault"
        )
        self.assertEqual(
            sigfault.GetChildMemberWithName("si_addr").GetValueAsUnsigned(), 0x1234
        )

        process.Continue()
        self.assertTrue(
            lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint).IsValid()
        )
        target = process.GetTarget()
        # 1 is SEGV_MAPERR.
        self.assertEqual(target.FindFirstGlobalVariable("g_code").GetValueAsSigned(), 1)
        self.assertEqual(
            target.FindFirstGlobalVariable("g_addr").GetValueAsUnsigned(), 0x1234
        )

    def test_expression(self):
        thread = self.launch()
        # Evaluate the expression before anyone asks for the stop description.
        result = thread.GetFrameAtIndex(0).EvaluateExpression("(int)getpid()")
        self.assertSuccess(result.GetError())
        self.check_siginfo(thread)

    def test_crashing_expression(self):
        thread = self.launch()
        result = thread.GetFrameAtIndex(0).EvaluateExpression("crash_at((int *)0x5678)")
        self.assertIn("fault address=0x5678", result.GetError().GetCString())
        self.check_siginfo(thread)
