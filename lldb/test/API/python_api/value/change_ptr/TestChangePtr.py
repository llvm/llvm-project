import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ChangePtrTest(TestBase):
    @skipIfWasm  # the test checks the address of a persistent expression result
    def test(self):
        self.build()

        src_file = lldb.SBFileSpec("main.c")
        _, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// break here 1", src_file
        )

        ## Test 1: The AddressOf of a dereferenced value should change when
        ## the pointer value is updated.

        frame = thread.GetFrameAtIndex(0)
        p = frame.FindVariable("p")
        deref = p.Dereference()
        self.assertEqual(deref.GetValueAsUnsigned(), 5)
        self.assertEqual(deref.AddressOf().GetValueAsUnsigned(), p.GetValueAsUnsigned())
        thread.StepOver()
        self.assertEqual(deref.GetValueAsUnsigned(), 7)
        self.assertEqual(deref.AddressOf().GetValueAsUnsigned(), p.GetValueAsUnsigned())

        ## Test 2: The AddressOf of a child value of a dereferenced value should
        ## change when the base pointer updates.

        lldbutil.continue_to_source_breakpoint(
            self, process, "// break here 2", src_file
        )
        frame = thread.GetFrameAtIndex(0)
        p = frame.FindVariable("p")
        deref_child = p.Dereference().GetChildMemberWithName("b")
        self.assertEqual(deref_child.GetValue(), "'b'")
        self.assertEqual(
            deref_child.AddressOf().GetValueAsUnsigned(), p.GetValueAsUnsigned() + 1
        )
        thread.StepOver()
        self.assertEqual(deref_child.GetValue(), "'d'")
        self.assertEqual(
            deref_child.AddressOf().GetValueAsUnsigned(), p.GetValueAsUnsigned() + 1
        )

        ## Test 3: Verify AddressOf updates correctly with persistent expression results.
        lldbutil.continue_to_source_breakpoint(
            self, process, "// break here 3", src_file
        )
        frame = thread.GetFrameAtIndex(0)
        frame.EvaluateExpression("int *$ptr = &a")
        ptr = frame.FindValue("$ptr", lldb.eValueTypeConstResult)
        deref = ptr.Dereference()
        self.assertEqual(deref.GetValueAsUnsigned(), 5)
        self.assertEqual(
            deref.AddressOf().GetValueAsUnsigned(),
            frame.FindVariable("a").AddressOf().GetValueAsUnsigned(),
        )
        frame.EvaluateExpression("$ptr = &b")
        self.assertEqual(deref.GetValueAsUnsigned(), 7)
        self.assertEqual(
            deref.AddressOf().GetValueAsUnsigned(),
            frame.FindVariable("b").AddressOf().GetValueAsUnsigned(),
        )
