"""
Test that nested types are parsed
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class CPPNestedTypeTestCase(TestBase):
    def test_with_run_command(self):
        """Test that nested types work in the expression evaluator"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// breakpoint 1", lldb.SBFileSpec("main.cpp")
        )

        self.expect_expr(
            "(int)PointerIntPairInfo::MaskAndShiftConstants::PointerBitMask",
            result_type="int",
            result_value="42",
        )

        self.expect_expr(
            "sizeof(PointerIntPairInfo::B)",
            result_type="unsigned long",
            result_value="1",
        )

        self.expect_expr(
            "sizeof(PointerIntPairInfo::C)",
            result_type="unsigned long",
            result_value="1",
        )
