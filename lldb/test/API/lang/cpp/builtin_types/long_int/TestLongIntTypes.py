import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def _long_size(self):
        """Return sizeof(long) in bytes, read from the 'long_size' global."""
        long_size = self.target().FindFirstGlobalVariable("long_size")
        self.assertTrue(long_size.IsValid(), "failed to read the 'long_size' global")
        return long_size.GetValueAsUnsigned()

    def test(self):
        """Check the types and values of all 'long'-typed variables."""
        self.build_and_run()

        # Check every scalar type both via 'frame variable' (var path) and via
        # the expression evaluator.
        self.expect_var_path("the_long", type="long", value="-1100110100")
        self.expect_expr("the_long", result_type="long", result_value="-1100110100")

        self.expect_var_path(
            "the_unsigned_long", type="unsigned long", value="1100110100"
        )
        self.expect_expr(
            "the_unsigned_long", result_type="unsigned long", result_value="1100110100"
        )

        self.expect_var_path("the_long_long", type="long long", value="-110011001100")
        self.expect_expr(
            "the_long_long", result_type="long long", result_value="-110011001100"
        )

        self.expect_var_path(
            "the_unsigned_long_long",
            type="unsigned long long",
            value="110011001100",
        )
        self.expect_expr(
            "the_unsigned_long_long",
            result_type="unsigned long long",
            result_value="110011001100",
        )

        # The min/max of 'long' depend on the data model. Zero and -1 are
        # width-independent.
        self.expect_var_path("long_zero", type="long", value="0")
        self.expect_var_path("long_neg_one", type="long", value="-1")
        self.expect_var_path("ulong_zero", type="unsigned long", value="0")

        self.expect_var_path(
            "llong_min", type="long long", value="-9223372036854775808"
        )
        self.expect_var_path("llong_max", type="long long", value="9223372036854775807")
        self.expect_var_path("llong_zero", type="long long", value="0")
        self.expect_var_path("llong_neg_one", type="long long", value="-1")
        self.expect_var_path("ullong_zero", type="unsigned long long", value="0")
        self.expect_var_path(
            "ullong_max",
            type="unsigned long long",
            value="18446744073709551615",
        )

        # Spot-check a few edge values through the expression evaluator too.
        self.expect_expr(
            "llong_min", result_type="long long", result_value="-9223372036854775808"
        )
        self.expect_expr(
            "ullong_max",
            result_type="unsigned long long",
            result_value="18446744073709551615",
        )

        # Check the min/max of 'long', whose width we read from the target rather
        # than inferring from the architecture name.
        long_size = self._long_size()
        if long_size == 8:
            self._check_long_lp64()
        elif long_size == 4:
            self._check_long_llp64()
        else:
            self.fail(
                "unexpected sizeof(long)=%d bytes; expected 4 (32-bit) or 8 "
                "(64-bit) long" % long_size
            )

    def _check_long_lp64(self):
        """Check the min/max of 'long' on LP64 targets (64-bit long)."""
        self.expect_var_path("long_min", type="long", value="-9223372036854775808")
        self.expect_var_path("long_max", type="long", value="9223372036854775807")
        self.expect_var_path(
            "ulong_max", type="unsigned long", value="18446744073709551615"
        )
        self.expect_expr(
            "long_min", result_type="long", result_value="-9223372036854775808"
        )
        self.expect_expr(
            "ulong_max",
            result_type="unsigned long",
            result_value="18446744073709551615",
        )

    def _check_long_llp64(self):
        """Check the min/max of 'long' on LLP64/ILP32 targets (32-bit long)."""
        self.expect_var_path("long_min", type="long", value="-2147483648")
        self.expect_var_path("long_max", type="long", value="2147483647")
        self.expect_var_path("ulong_max", type="unsigned long", value="4294967295")
        self.expect_expr("long_min", result_type="long", result_value="-2147483648")
        self.expect_expr(
            "ulong_max", result_type="unsigned long", result_value="4294967295"
        )
