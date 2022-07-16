# coding=utf8
"""
Test that C++ supports char8_t correctly.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class CxxChar8_tTestCase(TestBase):

    @skipIf(compiler="clang", compiler_version=['<', '7.0'])
    @expectedFailureDarwin(archs=["arm64", "arm64e"]) # <rdar://problem/37773624>
    def test_without_process(self):
        """Test that C++ supports char8_t without a running process."""
        self.build()
        lldbutil.run_to_breakpoint_make_target(self)

        self.expect("target variable a", substrs=["char8_t", "0x61 u8'a'"])
        self.expect("target variable ab",
                substrs=["const char8_t *", 'u8"你好"'])
        self.expect("target variable abc", substrs=["char8_t[9]", 'u8"你好"'])

        self.expect_expr("a", result_type="char8_t", result_summary="0x61 u8'a'")
        self.expect_expr("ab", result_type="const char8_t *", result_summary='u8"你好"')

        # FIXME: This should work too.
        self.expect("expr abc", substrs=['u8"你好"'], matching=False)


    @skipIf(compiler="clang", compiler_version=['<', '7.0'])
    def test_with_process(self):
        """Test that C++ supports char8_t with a running process."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, '// break here', lldb.SBFileSpec("main.cpp"))

        # As well as with it
        self.expect_expr("a", result_type="char8_t", result_summary="0x61 u8'a'")
        self.expect_expr("ab", result_type="const char8_t *", result_summary='u8"你好"')
        self.expect_expr("abc", result_type="char8_t[9]", result_summary='u8"你好"')
