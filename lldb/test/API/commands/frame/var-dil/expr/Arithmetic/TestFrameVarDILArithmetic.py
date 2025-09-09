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

        # Check unary results and integral promotion
        self.expect_var_path("+0", value="0")
        self.expect_var_path("-0", value="0")
        self.expect_var_path("+1", value="1")
        self.expect_var_path("-1", value="-1")
        self.expect_var_path("-9223372036854775808", value="9223372036854775808")
        self.expect_var_path("s", value="10", type="short")
        self.expect_var_path("+s", value="10", type="int")
        self.expect_var_path("-s", value="-10", type="int")
        self.expect_var_path("+us", value="1", type="int")
        self.expect_var_path("-us", value="-1", type="int")
        self.expect_var_path("+ref", value="2", type="int")
        self.expect_var_path("-ref", value="-2", type="int")
        self.expect_var_path("+0.0", value="0")
        self.expect_var_path("-0.0", value="-0")
        self.expect_var_path("+enum_one", value="1")
        self.expect_var_path("-enum_one", value="-1")
        self.expect_var_path("+wchar", value="1")
        self.expect_var_path("+char16", value="2")
        self.expect_var_path("+char32", value="3")
        self.expect_var_path("-bitfield.a", value="-1", type="int")
        self.expect_var_path("+bitfield.a", value="1", type="int")
        self.expect_var_path("+bitfield.b", value="2", type="int")
        self.expect_var_path("+bitfield.c", value="3", type="unsigned int")
        self.expect_var_path("+bitfield.d", value="4", type="uint64_t")
