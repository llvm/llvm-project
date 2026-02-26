"""
Test DIL BifField extraction.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarDILBitFieldExtraction(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_bitfield_extraction(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        # Test ranges and type
        self.expect_var_path("value[0:1]", value="3", type="int:2")
        self.expect_var_path("value[4:7]", value="7", type="int:4")
        self.expect_var_path("value[7:0]", value="115", type="int:8")

        # Test reference and dereferenced pointer
        self.expect_var_path("value_ref[0:1]", value="3", type="int:2")
        self.expect_var_path("(*value_ptr)[0:1]", value="3", type="int:2")

        # Test ranges as variable, reference, enum
        self.expect_var_path("value[idx_0:idx_1]", value="3", type="int:2")
        self.expect_var_path("value[0:idx_1_ref]", value="3", type="int:2")
        self.expect_var_path("value[idx_1_ref:0]", value="3", type="int:2")
        self.expect_var_path("value[0:enum_one]", value="3", type="int:2")
        self.expect_var_path("value[enum_one:0]", value="3", type="int:2")

        # Test array and pointer
        self.expect(
            "frame var 'int_arr[0:2]'",
            error=True,
            substrs=["bitfield range 0:2 is not valid"],
        )
        self.expect(
            "frame var 'value_ptr[0:1]'",
            error=True,
            substrs=["bitfield range 0:1 is not valid"],
        )

        # Test invalid input
        self.expect(
            "frame var 'value[1:]'",
            error=True,
            substrs=["Unexpected token: <']' (r_square)>"],
        )
        self.expect(
            "frame var 'value[1:2.0]'",
            error=True,
            substrs=["bit index is not an integer"],
        )
        self.expect(
            "frame var 'value[2.0:1]'",
            error=True,
            substrs=["bit index is not an integer"],
        )
        self.expect(
            "frame var 'value[0-2]'",
            error=True,
            substrs=["use of '-' for bitfield range is deprecated; use ':' instead"],
        )
