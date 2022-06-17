"""Test that we can unwind out of a signal handler.
   Which for AArch64 Linux requires a specific unwind plan."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class UnwindSignalTestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_unwind_signal(self):
        """Inferior calls sigill() and handles the resultant SIGILL.
           Stopped at a breakpoint in the handler, check that we can unwind
           back to sigill() and get the expected register contents there."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertState(process.GetState(), lldb.eStateStopped)
        signo = process.GetUnixSignals().GetSignalNumberFromName("SIGILL")

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonSignal)
        self.assertTrue(
            thread and thread.IsValid(),
            "Thread should be stopped due to a signal")
        self.assertTrue(
            thread.GetStopReasonDataCount() >= 1,
            "There should be data in the event.")
        self.assertEqual(thread.GetStopReasonDataAtIndex(0),
                         signo, "The stop signal should be SIGILL")

        # Continue to breakpoint in sigill handler
        bkpt = target.FindBreakpointByID(
            lldbutil.run_break_set_by_source_regexp(self, "Set a breakpoint here"))
        threads = lldbutil.continue_to_breakpoint(process, bkpt)
        self.assertEqual(len(threads), 1, "Expected single thread")
        thread = threads[0]

        # Expect breakpoint in 'handler'
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.GetDisplayFunctionName(), "handler", "Unexpected break?")

        # Expect that unwinding should find 'sigill'
        found_caller = False
        for frame in thread.get_thread_frames():
            if frame.GetDisplayFunctionName() == "sigill":
                # We should have ascending values in the x registers
                regs = frame.GetRegisters().GetValueAtIndex(0)
                err = lldb.SBError()

                for i in range(31):
                  name = 'x{}'.format(i)
                  value = regs.GetChildMemberWithName(name).GetValueAsUnsigned(err)
                  self.assertSuccess(err, "Failed to get register {}".format(name))
                  self.assertEqual(value, i, "Unexpected value for register {}".format(
                                      name))

                found_caller = True
                break

        self.assertTrue(found_caller, "Unwinding did not find func that caused the SIGILL")

        # Continue until we exit.
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)
