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
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
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

        # Check basic math and resulting types
        self.expect_var_path("1 + 2", value="3", type="int")
        self.expect_var_path("1 + true", value="2", type="int")
        self.expect_var_path("1L + wchar", value="2", type="long")
        self.expect_var_path("1L + char16", value="3", type="long")
        self.expect_var_path("1LL + char32", value="4", type="long long")
        self.expect_var_path("1UL + 1L", value="2", type="unsigned long")
        self.expect_var_path("s + x", value="12", type="int")
        self.expect_var_path("s + l", value="15", type="long")
        self.expect_var_path("1.0 + 2.5", value="3.5", type="double")
        self.expect_var_path("1 + 2.5f", value="3.5", type="float")
        self.expect_var_path("2. + .5", value="2.5", type="double")
        self.expect_var_path("2.f + .5f", value="2.5", type="float")
        self.expect_var_path("f + d", value="3.5", type="double")

        # Check limits and overflows
        frame = thread.GetFrameAtIndex(0)
        int_min = frame.GetValueForVariablePath("int_min").GetValue()
        int_max = frame.GetValueForVariablePath("int_max").GetValue()
        uint_max = frame.GetValueForVariablePath("uint_max").GetValue()
        ll_max = frame.GetValueForVariablePath("ll_max").GetValue()
        ll_min = frame.GetValueForVariablePath("ll_min").GetValue()
        ull_max = frame.GetValueForVariablePath("ull_max").GetValue()
        self.expect_var_path("int_max + 1", value=int_min)
        self.expect_var_path("uint_max + 1", value="0")
        self.expect_var_path("ll_max + 1", value=ll_min)
        self.expect_var_path("ull_max + 1", value="0")

        # Check references and typedefs
        self.expect_var_path("ref + 1", value="3")
        self.expect_var_path("my_ref + 1", value="3")
        self.expect_var_path("ref + my_ref", value="4")
