"""
Test evaluating expressions when debugging core file.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfLLVMTargetMissing("X86")
class CoreExprTestCase(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.target = self.dbg.CreateTarget("linux-x86_64.out")
        self.process = self.target.LoadCore("linux-x86_64.core")
        self.assertTrue(self.process, PROCESS_IS_VALID)

    def test_result_var(self):
        """Test that the result variable can be used in subsequent expressions."""

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

    def test_persist_var(self):
        """Test that user-defined variables can be used in subsequent expressions."""

        self.target.EvaluateExpression("int $my_int = 5")
        self.expect_expr("$my_int * 2", result_type="int", result_value="10")

    def test_context_object(self):
        """Test expression evaluation in context of an object."""

        val_outer = self.expect_expr("outer", result_type="Outer")

        val_inner = val_outer.EvaluateExpression("inner")
        self.assertTrue(val_inner.IsValid())
        self.assertEqual("Inner", val_inner.GetDisplayTypeName())

        val_val = val_inner.EvaluateExpression("this->val")
        self.assertTrue(val_val.IsValid())
        self.assertEqual("int", val_val.GetDisplayTypeName())
        self.assertEqual(val_val.GetValueAsSigned(), 5)
