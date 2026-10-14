"""
This tests that we do not lose control of the inferior, while doing an instruction-level step
over a thread creation instruction.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@requireThreadSupport
class CreateDuringInstructionStepTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @requireLinux
    @expectedFailureAndroid("llvm.org/pr24737", archs=["arm$"])
    @skipIf(oslist=["linux"], archs=["arm$", "aarch64"], bugnumber="llvm.org/pr24737")
    def test_step_inst(self):
        self.build()

        # This should create a breakpoint in the stepping thread.
        _, process, thread, _ = lldbutil.run_to_name_breakpoint(self, "main")

        # Make sure we see only one threads
        self.assertEqual(
            process.GetNumThreads(),
            1,
            "Number of expected threads and actual threads do not match.",
        )

        # Keep stepping until we see the thread creation
        while process.GetNumThreads() < 2:
            thread.StepInstruction(False)
            self.assertEqual(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
            self.assertEqual(
                thread.GetStopReason(),
                lldb.eStopReasonPlanComplete,
                "Step operation succeeded",
            )
            if self.TraceOn():
                self.runCmd("disassemble --pc")

        if self.TraceOn():
            self.runCmd("thread list")

        # We have successfully caught thread creation. Now just run to
        # completion
        process.Continue()

        # At this point, the inferior process should have exited.
        self.assertState(process.GetState(), lldb.eStateExited, PROCESS_EXITED)
