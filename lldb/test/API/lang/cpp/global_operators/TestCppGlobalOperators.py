"""
Test that global operators are found and evaluated.
"""
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCppGlobalOperators(TestBase):
    def prepare_executable_and_get_frame(self):
        self.build()

        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        return thread.GetSelectedFrame()

    def test_equals_operator(self):
        frame = self.prepare_executable_and_get_frame()

        test_result = frame.EvaluateExpression("operator==(s1, s2)")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValue() == "false",
            "operator==(s1, s2) = false",
        )

        test_result = frame.EvaluateExpression("operator==(s1, s3)")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValue() == "true",
            "operator==(s1, s3) = true",
        )

        test_result = frame.EvaluateExpression("operator==(s2, s3)")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValue() == "false",
            "operator==(s2, s3) = false",
        )

    def do_new_test(self, frame, expr, expected_value_name):
        """Evaluate a new expression, and check its result"""

        expected_value = frame.FindValue(
            expected_value_name, lldb.eValueTypeVariableGlobal
        )
        self.assertTrue(expected_value.IsValid())

        expected_value_addr = expected_value.AddressOf()
        self.assertTrue(expected_value_addr.IsValid())

        got = frame.EvaluateExpression(expr)
        self.assertTrue(got.IsValid())
        self.assertEqual(
            got.GetValueAsUnsigned(), expected_value_addr.GetValueAsUnsigned()
        )
        got_type = got.GetType()
        self.assertTrue(got_type.IsPointerType())
        self.assertEqual(got_type.GetPointeeType().GetName(), "Struct")

    @skipIfMTE  # Expression evaluation of overridden operator new fails under MTE.
    def test_operator_new(self):
        frame = self.prepare_executable_and_get_frame()

        self.do_new_test(frame, "new Struct()", "global_new_buf")
        self.do_new_test(frame, "new(new_tag) Struct()", "tagged_new_buf")
