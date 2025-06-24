"""
Test evaluating expressions when debugging core file.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfLLVMTargetMissing("X86")
class CoreExprTestCase(TestBase):
    def test_result_var(self):
        """Test that the result variable can be used in subsequent expressions."""

        target = self.dbg.CreateTarget("linux-x86_64.out")
        process = target.LoadCore("linux-x86_64.core")
        self.assertTrue(process, PROCESS_IS_VALID)

        self.expect_expr(
            "outer",
            result_type="Outer",
            result_children=[ValueCheck(name="inner", type="Inner")],
        )
        self.expect_expr(
            "$0.inner",
            result_type="Inner",
            result_children=[ValueCheck(name="val", type="int", value="5")],
        )
        self.expect_expr("$1.val", result_type="int", result_value="5")

    def test_context_object(self):
        """Tests expression evaluation in context of an object."""

        target = self.dbg.CreateTarget("linux-x86_64.out")
        process = target.LoadCore("linux-x86_64.core")
        self.assertTrue(process, PROCESS_IS_VALID)

        val_outer = self.expect_expr("outer", result_type="Outer")

        val_inner = val_outer.EvaluateExpression("inner")
        self.assertTrue(val_inner.IsValid())
        self.assertEqual("Inner", val_inner.GetDisplayTypeName())

        val_val = val_inner.EvaluateExpression("this->val")
        self.assertTrue(val_val.IsValid())
        self.assertEqual("int", val_val.GetDisplayTypeName())
        self.assertEqual(val_val.GetValueAsSigned(), 5)
