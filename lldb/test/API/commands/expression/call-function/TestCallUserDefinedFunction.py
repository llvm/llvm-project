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

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number(
            'main.cpp',
            '// Please test these expressions while stopped at this line:')

    def test(self):
        """Test return values of user defined function calls."""
        self.build()

        # Set breakpoint in main and run exe
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=-1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Test recursive function call.
        self.expect_expr("fib(5)", result_type="unsigned int", result_value="5")

        # Test function with more than one paramter
        self.expect_expr("add(4, 8)", result_type="int", result_value="12")

        # Test nesting function calls in function paramters
        self.expect_expr("add(add(5,2),add(3,4))", result_type="int", result_value="14")
        self.expect_expr("add(add(5,2),fib(5))", result_type="int", result_value="12")

        # Test function with pointer paramter
        self.expect_expr('stringCompare((const char*) \"Hello world\")', result_type="bool", result_value="true")
        self.expect_expr('stringCompare((const char*) \"Hellworld\")', result_type="bool", result_value="false")
