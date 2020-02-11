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
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # FIXME: We should be able to test this with target variable, but the
        # data formatter output is broken.
        lldbutil.run_break_set_by_symbol(self, 'main')
        self.runCmd("run", RUN_SUCCEEDED)

        self.expect_expr("a", result_type="char8_t", result_summary="0x61 u8'a'")
        self.expect_expr("ab", result_type="const char8_t *", result_summary='u8"你好"')
        self.expect_expr("abc", result_type="char8_t [9]", result_summary='u8"你好"')
