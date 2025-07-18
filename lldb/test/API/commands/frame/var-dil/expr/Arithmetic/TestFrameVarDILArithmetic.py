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
        self.expect_var_path("1", value="1", type="int")
        self.expect_var_path("1ull", value="1", type="unsigned long long")
        self.expect_var_path("0b10", value="2", type="int")
        self.expect_var_path("010", value="8", type="int")
        self.expect_var_path("0x10", value="16", type="int")
        self.expect_var_path("1.0", value="1", type="double")
        self.expect_var_path("1.0f", value="1", type="float")
        self.expect_var_path("0x1.2p+3f", value="9", type="float")
