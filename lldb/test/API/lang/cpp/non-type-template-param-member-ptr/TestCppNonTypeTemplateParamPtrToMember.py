import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test_member_data_pointer(self):
        """Member data pointer NTTPs: MemberData<&S::x> vs MemberData<&S::y>"""
        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        # Both must be resolvable as distinct specializations.
        self.expect_expr("md1", result_type="MemberData<&S::x>")
        self.expect_expr("md2", result_type="MemberData<&S::y>")
