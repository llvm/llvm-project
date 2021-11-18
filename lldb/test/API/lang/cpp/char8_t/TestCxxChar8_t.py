# coding=utf8
"""
Test that C++ supports char8_t correctly.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class CxxChar8_tTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIf(compiler="clang", compiler_version=['<', '7.0'])
    def test(self):
        """Test that C++ supports char8_t correctly."""
        self.build()

        lldbutil.run_to_breakpoint_make_target(self)

        # Make sure the variables can be printed without a running process.
        self.expect("target variable a", substrs=["char8_t", "0x61 u8'a'"])
        self.expect("target variable ab",
                substrs=["const char8_t *", 'u8"你好"'])
        self.expect("target variable abc", substrs=["char8_t[9]", 'u8"你好"'])

        self.expect_expr("a", result_type="char8_t", result_summary="0x61 u8'a'")
        self.expect_expr("ab", result_type="const char8_t *", result_summary='u8"你好"')
        # FIXME: This should work too.
        self.expect("expr abc", substrs=['u8"你好"'], matching=False)

        lldbutil.run_break_set_by_source_regexp(self, "// break here", "-f main.cpp")
        self.runCmd("run")

        # As well as with it
        self.expect_expr("a", result_type="char8_t", result_summary="0x61 u8'a'")
        self.expect_expr("ab", result_type="const char8_t *", result_summary='u8"你好"')
        self.expect_expr("abc", result_type="char8_t[9]", result_summary='u8"你好"')
