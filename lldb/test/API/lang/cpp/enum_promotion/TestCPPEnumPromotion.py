"""
Test LLDB type promotion of unscoped enums.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCPPEnumPromotion(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        UChar_promoted = self.frame().FindVariable("UChar_promoted")
        UShort_promoted = self.frame().FindVariable("UShort_promoted")
        UInt_promoted = self.frame().FindVariable("UInt_promoted")
        SLong_promoted = self.frame().FindVariable("SLong_promoted")
        ULong_promoted = self.frame().FindVariable("ULong_promoted")
        NChar_promoted = self.frame().FindVariable("NChar_promoted")
        NShort_promoted = self.frame().FindVariable("NShort_promoted")
        NInt_promoted = self.frame().FindVariable("NInt_promoted")
        NLong_promoted = self.frame().FindVariable("NLong_promoted")

        # Check that LLDB's promoted type is the same as the compiler's
        self.expect_expr("+EnumUChar::UChar", result_type=UChar_promoted.type.name)
        self.expect_expr("+EnumUShort::UShort", result_type=UShort_promoted.type.name)
        self.expect_expr("+EnumUInt::UInt", result_type=UInt_promoted.type.name)
        self.expect_expr("+EnumSLong::SLong", result_type=SLong_promoted.type.name)
        self.expect_expr("+EnumULong::ULong", result_type=ULong_promoted.type.name)
        self.expect_expr("+EnumNChar::NChar", result_type=NChar_promoted.type.name)
        self.expect_expr("+EnumNShort::NShort", result_type=NShort_promoted.type.name)
        self.expect_expr("+EnumNInt::NInt", result_type=NInt_promoted.type.name)
        self.expect_expr("+EnumNLong::NLong", result_type=NLong_promoted.type.name)
