"""Test that we can unwind out of a SIGABRT handler"""




import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class HandleAbortTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows  # signals do not exist on Windows
    @expectedFailureNetBSD
    def test_inferior_handle_sigabrt(self):
        """Inferior calls abort() and handles the resultant SIGABRT.
           Stopped at a breakpoint in the handler, verify that the backtrace
           includes the function that called abort()."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # launch
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetState(), lldb.eStateStopped)
        signo = process.GetUnixSignals().GetSignalNumberFromName("SIGABRT")

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonSignal)
        self.assertTrue(
            thread and thread.IsValid(),
            "Thread should be stopped due to a signal")
        self.assertTrue(
            thread.GetStopReasonDataCount() >= 1,
            "There should be data in the event.")
        self.assertEqual(thread.GetStopReasonDataAtIndex(0),
                         signo, "The stop signal should be SIGABRT")

        # Continue to breakpoint in abort handler
        bkpt = target.FindBreakpointByID(
            lldbutil.run_break_set_by_source_regexp(self, "Set a breakpoint here"))
        threads = lldbutil.continue_to_breakpoint(process, bkpt)
        self.assertEqual(len(threads), 1, "Expected single thread")
        thread = threads[0]

        # Expect breakpoint in 'handler'
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.GetDisplayFunctionName(), "handler", "Unexpected break?")

        # Expect that unwinding should find 'abort_caller'
        foundFoo = False
        for frame in thread:
            if frame.GetDisplayFunctionName() == "abort_caller":
                foundFoo = True

        self.assertTrue(foundFoo, "Unwinding did not find func that called abort")

        # Continue until we exit.
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)
