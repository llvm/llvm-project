import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )
        self.runCmd("command script import thing_formatter.py")
        frame = process.selected_thread.selected_frame
        x = frame.var("x")
        names = ("zero", "one")
        for i in range(x.num_children):
            idx = x.GetIndexOfChildWithName(f"[{i}]")
            self.assertEqual(idx, i)
            child = x.GetChildAtIndex(idx)
            self.assertEqual(child.name, names[idx])
        idx = x.GetIndexOfChildWithName(f"[{x.num_children + 1}]")
        self.assertEqual(idx, lldb.UINT32_MAX)
