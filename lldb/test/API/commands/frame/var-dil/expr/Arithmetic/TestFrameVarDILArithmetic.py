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

        # Check unary
        self.expect_var_path("+0", value="0")
        self.expect_var_path("-0", value="0")
        self.expect_var_path("+1", value="1")
        self.expect_var_path("-1", value="-1")
        self.expect_var_path("s", value="10")
        self.expect_var_path("+s", value="10")
        self.expect_var_path("-s", value="-10")
        self.expect_var_path("us", value="1")
        self.expect_var_path("-us", value="-1")
        self.expect_var_path("+0.0", value="0")
        self.expect_var_path("-0.0", value="-0")
        self.expect_var_path("-9223372036854775808", value="9223372036854775808")
        self.expect_var_path("+array", type="int *")
        self.expect_var_path("+enum_one", value="1")
        self.expect_var_path("-enum_one", value="4294967295") # TODO: fix
        self.expect_var_path("+bf.a", value="7")
        self.expect_var_path("+p", type="int *")
        self.expect(
            "frame var -- '-p'",
            error=True,
            substrs=["invalid argument type 'int *' to unary expression"],
        )
