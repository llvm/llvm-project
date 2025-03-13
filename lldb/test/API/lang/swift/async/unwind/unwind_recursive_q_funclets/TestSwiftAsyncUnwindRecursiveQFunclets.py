import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestCase(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    unwind_fail_range_cache = dict()

    @swiftTest
    @skipIf(oslist=["windows", "linux"])
    def test(self):
        """Test that the debugger can unwind at all instructions of all funclets"""
        self.build()

        source_file = lldb.SBFileSpec("main.swift")
        target, process, thread, bkpt = lldbutil.run_to_name_breakpoint(
            self, "$s1a9factorialyS2iYaFTQ1_"
        )

        # Ensure we are on the last factorial call which recurses (n == 1).
        frame = thread.frames[0]
        result = frame.EvaluateExpression("n == 1")
        self.assertSuccess(result.GetError())
        self.assertEqual(result.GetSummary(), "true")

        # Disable the breakpoint and step over the call.
        bkpt.SetEnabled(False)

        # Make sure we are still in the frame of n == 1.
        thread.StepOver()
        frame = thread.frames[0]
        result = frame.EvaluateExpression("n == 1")
        self.assertSuccess(result.GetError())
        self.assertEqual(result.GetSummary(), "true")

        thread.StepOver()
        frame = thread.frames[0]
        result = frame.EvaluateExpression("n == 1")
        self.assertSuccess(result.GetError())
        self.assertEqual(result.GetSummary(), "true")
