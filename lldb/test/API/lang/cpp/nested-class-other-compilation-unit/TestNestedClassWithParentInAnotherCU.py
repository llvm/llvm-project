"""
Test that the expression evaluator can access members of nested classes even if
the parents of the nested classes were imported from another compilation unit.
"""
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestNestedClassWithParentInAnotherCU(TestBase):

    def test_nested_class_with_parent_in_another_cu(self):
        self.main_source_file = lldb.SBFileSpec("main.cpp")
        self.build()
        (_, _, thread, _) = lldbutil.run_to_source_breakpoint(self, "// break here", self.main_source_file)
        frame = thread.GetSelectedFrame()
        # Parse the DIEs of the parent classes and the nested classes from
        # other.cpp's CU.
        warmup_result = frame.EvaluateExpression("b")
        self.assertTrue(warmup_result.IsValid())
        # Inspect fields of the nested classes. This will reuse the types that
        # were parsed during the evaluation above. By accessing a chain of
        # fields, we try to verify that all the DIEs, reused types and
        # declaration contexts were wired properly into lldb's parser's state.
        expr_result = frame.EvaluateExpression("a.y.oY_inner.oX_inner")
        self.assertTrue(expr_result.IsValid())
        self.assertEqual(expr_result.GetValue(), "42")
