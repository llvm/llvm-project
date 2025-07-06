"""
Test DIL BifField extraction.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarDILBitFieldExtraction(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def expect_var_path(self, expr, compare_to_framevar=False, value=None, type=None):
        value_dil = super().expect_var_path(expr, value=value, type=type)
        if compare_to_framevar:
            self.runCmd("settings set target.experimental.use-DIL false")
            value_frv = super().expect_var_path(expr, value=value, type=type)
            self.runCmd("settings set target.experimental.use-DIL true")
            self.assertEqual(value_dil.GetValue(), value_frv.GetValue())

    def test_bitfield_extraction(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        # Test ranges and type
        self.expect_var_path("value[0-1]", True, value="3", type="int:2")
        self.expect_var_path("value[4-7]", True, value="7", type="int:4")
        self.expect_var_path("value[7-0]", True, value="115", type="int:8")

        # Test reference and dereferenced pointer
        self.expect_var_path("value_ref[0-1]", value="3", type="int:2")
        self.expect_var_path("(*value_ptr)[0-1]", value="3", type="int:2")

        # Test array and pointer
        self.expect(
            "frame var 'int_arr[0-2]'",
            error=True,
            substrs=["bitfield range 0-2 is not valid"],
        )
        self.expect(
            "frame var 'value_ptr[0-1]'",
            error=True,
            substrs=["bitfield range 0-1 is not valid"],
        )

        # Test invalid input
        self.expect(
            "frame var 'value[1-]'",
            error=True,
            substrs=["failed to parse integer constant"],
        )
