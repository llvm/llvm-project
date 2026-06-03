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

        # Check errors
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
