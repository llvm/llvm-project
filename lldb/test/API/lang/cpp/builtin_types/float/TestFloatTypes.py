import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        """Check the types and values of all float-typed variables.
        NOTE: TestFloatDisplay.py tests displaying of the various float values
        and we only test that we can recognize the different types and can
        extract their values correctly.
        """
        self.build_and_run()

        # Check every scalar floating point type both via 'frame variable' (var
        # path) and via the expression evaluator.
        self.expect_var_path("the_float", type="float", value="3.5")
        self.expect_expr("the_float", result_type="float", result_value="3.5")

        self.expect_var_path("the_double", type="double", value="6.25")
        self.expect_expr("the_double", result_type="double", result_value="6.25")

        self.expect_var_path("the_long_double", type="long double", value="10.75")
        self.expect_expr(
            "the_long_double", result_type="long double", result_value="10.75"
        )

        # Check edge-case values: zero, -1 and a negative value.
        self.expect_var_path("float_zero", type="float", value="0")
        self.expect_var_path("float_neg_one", type="float", value="-1")
        self.expect_var_path("float_neg", type="float", value="-2.5")
        self.expect_var_path("double_zero", type="double", value="0")
        self.expect_var_path("double_neg_one", type="double", value="-1")
        self.expect_var_path("double_neg", type="double", value="-2.5")
        self.expect_var_path("long_double_zero", type="long double", value="0")
        self.expect_var_path("long_double_neg_one", type="long double", value="-1")

        self.expect_expr("float_neg", result_type="float", result_value="-2.5")
        self.expect_expr("double_zero", result_type="double", result_value="0")
        self.expect_expr(
            "long_double_neg_one", result_type="long double", result_value="-1"
        )
