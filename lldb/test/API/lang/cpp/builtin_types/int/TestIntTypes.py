import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        """Check the types and values of all integer-typed variables."""
        self.build_and_run()

        self.expect_var_path("the_short", type="short", value="-31987")
        self.expect_expr("the_short", result_type="short", result_value="-31987")

        self.expect_var_path("the_unsigned_short", type="unsigned short", value="65000")
        self.expect_expr(
            "the_unsigned_short", result_type="unsigned short", result_value="65000"
        )

        self.expect_var_path("the_int", type="int", value="-1100110")
        self.expect_expr("the_int", result_type="int", result_value="-1100110")

        self.expect_var_path(
            "the_unsigned_int", type="unsigned int", value="4000000000"
        )
        self.expect_expr(
            "the_unsigned_int", result_type="unsigned int", result_value="4000000000"
        )

        # Check edge-case values: smallest, largest, zero and -1.
        self.expect_var_path("short_min", type="short", value="-32768")
        self.expect_var_path("short_max", type="short", value="32767")
        self.expect_var_path("short_zero", type="short", value="0")
        self.expect_var_path("short_neg_one", type="short", value="-1")
        self.expect_var_path("ushort_zero", type="unsigned short", value="0")
        self.expect_var_path("ushort_max", type="unsigned short", value="65535")

        self.expect_var_path("int_min", type="int", value="-2147483648")
        self.expect_var_path("int_max", type="int", value="2147483647")
        self.expect_var_path("int_zero", type="int", value="0")
        self.expect_var_path("int_neg_one", type="int", value="-1")
        self.expect_var_path("uint_zero", type="unsigned int", value="0")
        self.expect_var_path("uint_max", type="unsigned int", value="4294967295")
