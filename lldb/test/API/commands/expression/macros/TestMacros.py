

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestMacros(TestBase):

    @skipIf(compiler="clang", compiler_version=['<', '9.0'])
    @expectedFailureAll(
        compiler="clang",
        bugnumber="clang does not emit .debug_macro[.dwo] sections.")
    @expectedFailureAll(
        debug_info="dwo",
        bugnumber="GCC produces multiple .debug_macro.dwo sections and the spec is unclear as to what it means")
    @expectedFailureAll(
        hostoslist=["windows"],
        compiler="gcc",
        triple='.*-android')
    @expectedFailureAll(
        compiler="gcc", compiler_version=['<', '5.1'],
        bugnumber=".debug_macro was introduced in DWARF 5, GCC supports it since version 5.1")
    def test_expr_with_macros(self):
        self.build()

        # Get main source file
        src_file = "main.cpp"
        hdr_file = "macro1.h"
        src_file_spec = lldb.SBFileSpec(src_file)
        self.assertTrue(src_file_spec.IsValid(), "Main source file")

        (target, process, thread, bp1) = lldbutil.run_to_source_breakpoint(
            self, "Break here", src_file_spec)

        # Get frame for current thread
        frame = thread.GetSelectedFrame()

        result = frame.EvaluateExpression("MACRO_1")
        self.assertTrue(
            result.IsValid() and result.GetValue() == "100",
            "MACRO_1 = 100")

        result = frame.EvaluateExpression("MACRO_2")
        self.assertTrue(
            result.IsValid() and result.GetValue() == "200",
            "MACRO_2 = 200")

        result = frame.EvaluateExpression("ONE")
        self.assertTrue(
            result.IsValid() and result.GetValue() == "1",
            "ONE = 1")

        result = frame.EvaluateExpression("TWO")
        self.assertTrue(
            result.IsValid() and result.GetValue() == "2",
            "TWO = 2")

        result = frame.EvaluateExpression("THREE")
        self.assertTrue(
            result.IsValid() and result.GetValue() == "3",
            "THREE = 3")

        result = frame.EvaluateExpression("FOUR")
        self.assertTrue(
            result.IsValid() and result.GetValue() == "4",
            "FOUR = 4")

        result = frame.EvaluateExpression("HUNDRED")
        self.assertTrue(
            result.IsValid() and result.GetValue() == "100",
            "HUNDRED = 100")

        result = frame.EvaluateExpression("THOUSAND")
        self.assertTrue(
            result.IsValid() and result.GetValue() == "1000",
            "THOUSAND = 1000")

        result = frame.EvaluateExpression("MILLION")
        self.assertTrue(result.IsValid() and result.GetValue()
                        == "1000000", "MILLION = 1000000")

        result = frame.EvaluateExpression("MAX(ONE, TWO)")
        self.assertTrue(
            result.IsValid() and result.GetValue() == "2",
            "MAX(ONE, TWO) = 2")

        result = frame.EvaluateExpression("MAX(THREE, TWO)")
        self.assertTrue(
            result.IsValid() and result.GetValue() == "3",
            "MAX(THREE, TWO) = 3")

        # Get the thread of the process
        thread.StepOver()

        # Get frame for current thread
        frame = thread.GetSelectedFrame()

        result = frame.EvaluateExpression("MACRO_2")
        self.assertTrue(
            result.GetError().Fail(),
            "Printing MACRO_2 fails in the mail file")

        result = frame.EvaluateExpression("FOUR")
        self.assertTrue(
            result.GetError().Fail(),
            "Printing FOUR fails in the main file")

        thread.StepInto()

        # Get frame for current thread
        frame = thread.GetSelectedFrame()

        result = frame.EvaluateExpression("ONE")
        self.assertTrue(
            result.IsValid() and result.GetValue() == "1",
            "ONE = 1")

        result = frame.EvaluateExpression("MAX(ONE, TWO)")
        self.assertTrue(
            result.IsValid() and result.GetValue() == "2",
            "MAX(ONE, TWO) = 2")

        # This time, MACRO_1 and MACRO_2 are not visible.
        result = frame.EvaluateExpression("MACRO_1")
        self.assertTrue(result.GetError().Fail(),
                        "Printing MACRO_1 fails in the header file")

        result = frame.EvaluateExpression("MACRO_2")
        self.assertTrue(result.GetError().Fail(),
                        "Printing MACRO_2 fails in the header file")

        # Check that the macro definitions do not trigger bogus Clang
        # diagnostics about macro redefinitions.
        result = frame.EvaluateExpression("does_not_parse")
        self.assertNotIn("macro redefined", str(result.GetError()))
        self.assertNotIn("redefining builtin macro", str(result.GetError()))
