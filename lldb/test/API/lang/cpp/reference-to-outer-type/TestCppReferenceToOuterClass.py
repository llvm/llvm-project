import unittest
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @unittest.expectedFailure  # The fix for this was reverted due to llvm.org/PR52257
    def test(self):
        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        test_var = self.expect_expr("test_var", result_type="In")
        nested_member = test_var.GetChildMemberWithName("NestedClassMember")
        self.assertEqual("Outer::NestedClass", nested_member.GetType().GetName())
