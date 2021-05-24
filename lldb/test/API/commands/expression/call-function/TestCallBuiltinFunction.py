"""
Tests calling builtin functions using expression evaluation.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprCommandCallBuiltinFunction(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # Builtins are expanded by Clang, so debug info shouldn't matter.
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()

        target = self.createTestTarget()

        self.expect_expr("__builtin_isinf(0.0f)", result_type="int", result_value="0")
        self.expect_expr("__builtin_isnormal(0.0f)", result_type="int", result_value="0")
        self.expect_expr("__builtin_constant_p(1)", result_type="int", result_value="1")
        self.expect_expr("__builtin_abs(-14)", result_type="int", result_value="14")
