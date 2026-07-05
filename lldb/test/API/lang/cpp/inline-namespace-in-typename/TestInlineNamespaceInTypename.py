import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestInlineNamespaceInTypename(TestBase):
    def test(self):
        """
        Tests that we correctly omit the inline namespace when printing
        the type name for "display", even if omitting the inline namespace
        would be ambiguous in the current context.
        """
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))

        t1 = target.FindGlobalVariables("t1", 1)
        self.assertTrue(len(t1), 1)
        self.assertEqual(t1[0].GetDisplayTypeName(), "foo::Duplicate")

        # 'foo::Duplicate' would be an ambiguous reference, but we still
        # omit the inline namespace when displaying the type.
        t2 = target.FindGlobalVariables("t2", 1)
        self.assertTrue(len(t2), 1)
        self.assertEqual(t2[0].GetDisplayTypeName(), "foo::Duplicate")
        self.assertEqual(t2[0].GetTypeName(), "foo::bar::Duplicate")

        t3 = target.FindGlobalVariables("t3", 1)
        self.assertTrue(len(t3), 1)
        self.assertEqual(t3[0].GetDisplayTypeName(), "foo::Unique")
        self.assertEqual(t3[0].GetTypeName(), "foo::bar::Unique")
