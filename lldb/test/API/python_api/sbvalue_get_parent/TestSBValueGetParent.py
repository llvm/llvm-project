import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )
        frame = thread.GetFrameAtIndex(0)

        parent = frame.FindVariable("parent")
        self.assertTrue(parent.IsValid())

        child = parent.GetChildMemberWithName("child")
        self.assertTrue(child.IsValid())

        # GetParent of child should be the parent struct.
        child_parent = child.GetParent()
        self.assertTrue(child_parent.IsValid())
        self.assertEqual(child_parent.name, "parent")
        self.assertEqual(child_parent.GetID(), parent.GetID())

        # GetParent of a top-level variable should be invalid.
        self.assertFalse(parent.GetParent().IsValid())
