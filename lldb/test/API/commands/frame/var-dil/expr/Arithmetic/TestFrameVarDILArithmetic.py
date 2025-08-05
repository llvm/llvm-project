"""
Test DIL arithmetic.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarDILArithmetic(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_arithmetic(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        # Check number parsing
        self.expect_var_path("1.0", value="1", type="double")
        self.expect_var_path("1.0f", value="1", type="float")
        self.expect_var_path("0x1.2p+3f", value="9", type="float")
        self.expect_var_path("1", value="1", type="int")
        self.expect_var_path("1u", value="1", type="unsigned int")
        self.expect_var_path("0b1l", value="1", type="long")
        self.expect_var_path("01ul", value="1", type="unsigned long")
        self.expect_var_path("0o1ll", value="1", type="long long")
        self.expect_var_path("0x1ULL", value="1", type="unsigned long long")
        self.expect_var_path("0xFFFFFFFFFFFFFFFF", value="18446744073709551615")
        self.expect(
            "frame var '0xFFFFFFFFFFFFFFFFF'",
            error=True,
            substrs=[
                "integer literal is too large to be represented in any integer type"
            ],
        )
