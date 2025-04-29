import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


@skipIf(oslist=["windows", "linux"])
@skipIfAsan
class TestSwiftSteppingThroughWitness(TestBase):

    def setUp(self):
        TestBase.setUp(self)
        self.runCmd(
            "settings set target.process.thread.step-avoid-libraries libswift_Concurrency.dylib"
        )

    @swiftTest
    def test_step_in_and_out(self):
        """Test that stepping in and out of protocol methods work"""
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        thread.StepInto()
        stop_reason = thread.GetStopReason()
        self.assertStopReason(stop_reason, lldb.eStopReasonPlanComplete)

        frame0 = thread.frames[0]
        frame1 = thread.frames[1]
        self.assertIn("SlowRandomNumberGenerator.random", frame0.GetFunctionName())
        self.assertIn(
            "protocol witness for a.RandomNumberGenerator.random",
            frame1.GetFunctionName(),
        )

        thread.StepOut()
        stop_reason = thread.GetStopReason()
        self.assertStopReason(stop_reason, lldb.eStopReasonPlanComplete)
        frame0 = thread.frames[0]
        self.assertIn("doMath", frame0.GetFunctionName())

    @swiftTest
    def test_step_over(self):
        """Test that stepping over protocol methods work"""
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        thread.StepOver()
        stop_reason = thread.GetStopReason()
        self.assertStopReason(stop_reason, lldb.eStopReasonPlanComplete)
        frame0 = thread.frames[0]
        self.assertIn("doMath", frame0.GetFunctionName())

        line_entry = frame0.GetLineEntry()
        self.assertEqual(14, line_entry.GetLine())
