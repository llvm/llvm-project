"""
Test calling user defined functions using expression evaluation.

Note:
  LLDBs current first choice of evaluating functions is using the IR interpreter,
  which is only supported on Hexagon. Otherwise JIT is used for the evaluation.

"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprCommandCallUserDefinedFunction(TestBase):
    def test(self):
        """Test return values of user defined function calls."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        # Test recursive function call.
        self.expect_expr("fib(5)", result_type="unsigned int", result_value="5")

        # Test function with more than one parameter
        self.expect_expr("add(4, 8)", result_type="int", result_value="12")

        # Test nesting function calls in function parameters
        self.expect_expr("add(add(5,2),add(3,4))", result_type="int", result_value="14")
        self.expect_expr("add(add(5,2),fib(5))", result_type="int", result_value="12")

        # Test function with pointer parameter
        self.expect_expr(
            'stringCompare((const char*) "Hello world")',
            result_type="bool",
            result_value="true",
        )
        self.expect_expr(
            'stringCompare((const char*) "Hellworld")',
            result_type="bool",
            result_value="false",
        )
