import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from typing import Union


class TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )
        frame = thread.GetFrameAtIndex(0)

        self._do_test(frame, "parent", "child")
        self._do_test(frame, "vec", 0)

        # Test Python synthetic formatters using CreateChildAtOffset.
        self.runCmd("command script import MyContainer_synthetic.py")
        self._do_test(frame, "container", 0)

    def _do_test(
        self, frame: lldb.SBFrame, parent_name: str, child_key: Union[str, int]
    ):
        parent = frame.FindVariable(parent_name)
        self.assertTrue(parent.IsValid())

        if isinstance(child_key, int):
            child = parent.GetChildAtIndex(child_key)
        else:
            assert isinstance(child_key, str)
            child = parent.GetChildMemberWithName(child_key)
        self.assertTrue(child.IsValid())

        # GetParent of child should be the original parent.
        child_parent = child.GetParent()
        self.assertTrue(child_parent.IsValid())
        self.assertEqual(child_parent.name, parent_name)
        self.assertEqual(child_parent.GetID(), parent.GetID())
