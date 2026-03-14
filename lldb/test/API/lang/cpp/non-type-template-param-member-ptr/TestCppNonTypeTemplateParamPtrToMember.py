import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    def test_member_data_pointer(self):
        """Member data pointer NTTPs: MemberData<&S::x> vs MemberData<&S::y>"""
        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        # Both must be resolvable as distinct specializations.
        # Without the fix, md2 fails with "undeclared identifier".
        self.expect_expr("md1")
        self.expect_expr("md2")

    @no_debug_info_test
    def test_nullptr(self):
        """nullptr NTTP: MaybeNull<nullptr> vs MaybeNull<&g1>"""
        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        # nullptr (const_value=0) vs pointer (DW_AT_location) produce
        # different TemplateArgument kinds, so they are naturally distinct.
        self.expect_expr("mn1")
        self.expect_expr("mn2")
