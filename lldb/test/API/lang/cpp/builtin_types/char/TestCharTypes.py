import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build_and_run()

        self.expect_var_path("the_char", type="char", value="'a'")
        self.expect_expr("the_char", result_type="char", result_value="'a'")

        self.expect_var_path("the_signed_char", type="signed char", value="'B'")
        self.expect_expr(
            "the_signed_char", result_type="signed char", result_value="'B'"
        )

        self.expect_var_path("the_unsigned_char", type="unsigned char", value="'Z'")
        self.expect_expr(
            "the_unsigned_char", result_type="unsigned char", result_value="'Z'"
        )

        # Check edge-case values: smallest, largest, zero and -1.
        self.expect_var_path("char_zero", type="char", value="'\\0'")
        self.expect_var_path("char_neg_one", type="char", value="'\\xff'")
        self.expect_var_path("char_high_bit", type="char", value="'\\x80'")
        self.expect_var_path("char_low_max", type="char", value="'\\x7f'")
        self.expect_var_path("schar_neg_one", type="signed char", value="'\\xff'")
        self.expect_var_path("schar_min", type="signed char", value="'\\x80'")
        self.expect_var_path("schar_max", type="signed char", value="'\\x7f'")
        self.expect_var_path("uchar_zero", type="unsigned char", value="'\\0'")
        self.expect_var_path("uchar_max", type="unsigned char", value="'\\xff'")