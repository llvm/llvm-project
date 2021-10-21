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
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.cpp"))

        self.expect_expr("a", result_type="char8_t", result_summary="0x61 u8'a'")
        self.expect_expr("ab", result_type="const char8_t *", result_summary='u8"你好"')
        self.expect_expr("abc", result_type="char8_t[9]", result_summary='u8"你好"')
