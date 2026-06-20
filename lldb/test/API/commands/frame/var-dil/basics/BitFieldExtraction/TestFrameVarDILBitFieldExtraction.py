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

        # Test that old range syntax is now a binary subtraction
        self.expect_var_path("value[6-1]", value="1", type="int:1")

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

        # A negative bit index must be rejected with a clear error instead of
        # silently wrapping to a huge uint32_t at the GetSyntheticBitFieldChild
        # call site.
        self.expect(
            "frame var 'value[-1:0]'",
            error=True,
            substrs=["bitfield range -1:0 is not valid (negative index)"],
        )
        self.expect(
            "frame var 'value[0:-1]'",
            error=True,
            substrs=["bitfield range 0:-1 is not valid (negative index)"],
        )

        # A bitfield wider than 64 bits must be rejected. The underlying
        # DataExtractor::GetMaxU64Bitfield only supports up to 64 bits
        # (it asserts and otherwise performs an out-of-bounds shift).
        self.expect(
            "frame var 'value[0:64]'",
            error=True,
            substrs=["bitfield range 0:64 is not valid (more than 64 bits)"],
        )

        # A bitfield whose high index is past the base object's storage must be
        # rejected. Otherwise reading/formatting the synthetic child performs an
        # out-of-bounds shift in DataExtractor::GetMaxU64Bitfield. 'value' is a
        # 32-bit int, so bit index 50 is out of range. The range is normalized
        # (high:low swapped) before the message is built.
        self.expect(
            "frame var 'value[100:50]'",
            error=True,
            substrs=["bitfield range 50:100 is not valid"],
        )
