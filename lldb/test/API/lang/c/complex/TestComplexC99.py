import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    @skipIfWindows
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )

        self.expect_expr(
            "complex_float", result_type="_Complex float", result_value="-1.5 + -2.5i"
        )
        self.expect_expr(
            "complex_float + (2.0f + 3.5fi)",
            result_type="_Complex float",
            result_value="0.5 + 1i",
        )

        self.expect_expr(
            "complex_double", result_type="_Complex double", result_value="-1.5 + -2.5i"
        )
        self.expect_expr(
            "complex_double + (2.0 + 3.5i)",
            result_type="_Complex double",
            result_value="0.5 + 1i",
        )

    @no_debug_info_test
    # FIXME: LLDB fails to read the imaginary part of the number.
    @expectedFailureAll()
    @skipIfWindows
    def test_long_double(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )

        self.expect_expr(
            "complex_long_double",
            result_type="_Complex long double",
            result_value="-1.5 + 1i",
        )
        self.expect_expr(
            "complex_long_double + (2.0L + 3.5Li)",
            result_type="_Complex long double",
            result_value="0.5 + 1i",
        )
