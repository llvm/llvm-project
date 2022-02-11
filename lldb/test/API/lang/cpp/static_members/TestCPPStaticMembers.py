"""
Tests that C++ member and static variables have correct layout and scope.
"""



import unittest2
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # We fail to lookup static members on Windows.
    @expectedFailureAll(oslist=["windows"])
    def test_access_from_main(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// stop in main", lldb.SBFileSpec("main.cpp"))

        self.expect_expr("my_a.m_a", result_type="short", result_value="1")
        self.expect_expr("my_a.s_b", result_type="long", result_value="2")
        self.expect_expr("my_a.s_c", result_type="int", result_value="3")

    # We fail to lookup static members on Windows.
    @expectedFailureAll(oslist=["windows"])
    def test_access_from_member_function(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// stop in member function", lldb.SBFileSpec("main.cpp"))
        self.expect_expr("m_a", result_type="short", result_value="1")
        self.expect_expr("s_b", result_type="long", result_value="2")
        self.expect_expr("s_c", result_type="int", result_value="3")

    # Currently lookups find variables that are in any scope.
    @expectedFailureAll()
    def test_access_without_scope(self):
        self.build()
        self.createTestTarget()
        self.expect("expression s_c", error=True,
                    startstr="error: use of undeclared identifier 's_d'")

    # We fail to lookup static members on Windows.
    @expectedFailureAll(oslist=["windows"])
    def test_no_crash_in_IR_arithmetic(self):
        """
        Test that LLDB doesn't crash on evaluating specific expression involving
        pointer arithmetic and taking the address of a static class member.
        See https://bugs.llvm.org/show_bug.cgi?id=52449
        """
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// stop in main", lldb.SBFileSpec("main.cpp"))

        # This expression contains the following IR code:
        # ... i64 ptrtoint (i32* @_ZN1A3s_cE to i64)) ...
        expr = "(int*)100 + (long long)(&A::s_c)"

        # The IR interpreter doesn't support non-const operands to the
        # `GetElementPtr` IR instruction, so verify that it correctly fails to
        # evaluate expression.
        opts = lldb.SBExpressionOptions()
        opts.SetAllowJIT(False)
        value = self.target().EvaluateExpression(expr, opts)
        self.assertTrue(value.GetError().Fail())
        self.assertIn(
            "Can't evaluate the expression without a running target",
            value.GetError().GetCString())

        # Evaluating the expression via JIT should work fine.
        value = self.target().EvaluateExpression(expr)
        self.assertSuccess(value.GetError())
