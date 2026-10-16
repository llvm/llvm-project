"""
Test DIL bitwise operators.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarDILBitwise(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_bitwise(self):
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        # Check unary negation
        self.expect_var_path("~(-1)", value="0")
        self.expect_var_path("~~0", value="0")
        self.expect_var_path("~0", value="-1")
        self.expect_var_path("~1", value="-2")
        self.expect_var_path("~0LL", value="-1")
        self.expect_var_path("~1LL", value="-2")
        self.expect_var_path("~true", value="-2")
        self.expect_var_path("~false", value="-1")
        self.expect_var_path("~var_true", value="-2")
        self.expect_var_path("~var_false", value="-1")
        self.expect_var_path("~ull_max", value="0")
        self.expect_var_path("~0b1011", value="-12")
        self.expect_var_path("~sh", value="-2")
        self.expect_var_path("~sh_ref", value="-2")

        # Check bitwise shifts
        self.expect_var_path("(1 << 5)", value="32")
        self.expect_var_path("(32 >> 2)", value="8")
        self.expect_var_path("(-1 >> 10)", value="-1")
        self.expect_var_path("(-100 >> 5)", value="-4")
        self.expect_var_path("(-3 << 6)", value="-192")
        self.expect_var_path("(-1 >> 1U)", value="-1")
        self.expect_var_path("(0xFFFFFFFFu>>31)", value="1")
        self.expect_var_path("(char)1 << 16", value="65536")
        self.expect_var_path("(signed char)-123 >> 8", value="-1")
        self.expect_var_path("enum_one << enum_one", value="2")
        self.expect_var_path("2 >> enum_one", value="1")
        self.expect_var_path("i64 << 63", type="uint64_t")

        # Check And, Xor, Or
        self.expect_var_path("0b1011 & 0xFF", value="11")
        self.expect_var_path("0b1011 & mask_ff", value="11")
        self.expect_var_path("0b1011 & 0b0111", value="3")
        self.expect_var_path("0b1011 | 0b0111", value="15")
        self.expect_var_path("-0b1011 | 0xFF", value="-1")
        self.expect_var_path("-0b1011 | 0xFFu", value="4294967295")
        self.expect_var_path("0b1011 ^ 0b0111", value="12")

        # Check errors
        self.expect(
            "frame var -- '~1.0'",
            error=True,
            substrs=["invalid argument type 'double' to unary expression"],
        )
        self.expect(
            "frame var -- '~s'",
            error=True,
            substrs=["invalid argument type 'S' to unary expression"],
        )
        self.expect(
            "frame var -- 's & 1.0'",
            error=True,
            substrs=["invalid operands to binary expression ('S' and 'double')"],
        )
        self.expect(
            "frame var -- '1 ^ s'",
            error=True,
            substrs=["invalid operands to binary expression ('int' and 'S')"],
        )
        self.expect(
            "frame var -- '1 | 1.0'",
            error=True,
            substrs=["invalid operands to binary expression ('int' and 'double')"],
        )
        self.expect(
            "frame var -- '1 << 1.0'",
            error=True,
            substrs=["invalid operands to binary expression ('int' and 'double')"],
        )
        self.expect(
            "frame var -- 's << 1'",
            error=True,
            substrs=["invalid operands to binary expression ('S' and 'int')"],
        )
        self.expect(
            "frame var -- '1 >> -1'",
            error=True,
            substrs=["invalid shift amount"],
        )
        self.expect(
            "frame var -- 'i64 << 64'",
            error=True,
            substrs=["invalid shift amount"],
        )

        # Check that bitwise & is allowed only in full mode
        frame = thread.GetFrameAtIndex(0)
        simple = frame.GetValueForVariablePathWithMode("i64 & 1", lldb.eDILModeSimple)
        legacy = frame.GetValueForVariablePathWithMode("i64 & 1", lldb.eDILModeLegacy)
        legacy_other = frame.var_with_mode("i64 & 1", lldb.eDILModeLegacy)
        full = frame.GetValueForVariablePathWithMode("i64 & 1", lldb.eDILModeFull)
        full_other = frame.var_with_mode("i64 & 1", lldb.eDILModeFull)
        self.assertFailure(simple.GetError())
        self.assertFailure(legacy.GetError())
        self.assertFailure(legacy_other.GetError())
        self.assertSuccess(full.GetError())
        self.assertSuccess(full_other.GetError())
