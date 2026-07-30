import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ChangePtrTest(TestBase):
    def test(self):
        self.build()

        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )
        frame = thread.GetFrameAtIndex(0)
        p = frame.FindVariable("p")
        deref = p.Dereference()
        self.assertEqual(deref.GetValueAsUnsigned(), 5)
        self.assertEqual(deref.AddressOf().GetValueAsUnsigned(), p.GetValueAsUnsigned())
        thread.StepOver()
        self.assertEqual(deref.GetValueAsUnsigned(), 7)
        self.assertEqual(deref.AddressOf().GetValueAsUnsigned(), p.GetValueAsUnsigned())
