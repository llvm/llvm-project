import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftClosureTypes(TestBase):
    @swiftTest
    def test(self):
        self.build()

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.frames[0]

        self.assertEqual(frame.FindVariable("f0_0").GetDisplayTypeName(), "() -> ()")
        self.assertEqual(frame.FindVariable("f0_1").GetDisplayTypeName(), "() -> Int")
        self.assertEqual(frame.FindVariable("f0_2").GetDisplayTypeName(), "() -> (Int, Double)")

        self.assertEqual(frame.FindVariable("f1_0").GetDisplayTypeName(), "(Float) -> ()")
        self.assertEqual(frame.FindVariable("f1_1").GetDisplayTypeName(), "(Int) -> Double")
        self.assertEqual(frame.FindVariable("f1_2").GetDisplayTypeName(), "(Double) -> (Float, Int)")

        self.assertEqual(frame.FindVariable("f2_0").GetDisplayTypeName(), "(Int, Double) -> ()")
        self.assertEqual(frame.FindVariable("f2_1").GetDisplayTypeName(), "(Float, Int) -> Double")
        self.assertEqual(frame.FindVariable("f2_2").GetDisplayTypeName(), "(Int, Float) -> (Double, Int)")
