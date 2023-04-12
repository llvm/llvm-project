"""Test that stepping in/out of trampolines works as expected.
"""



from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestTrampoline(TestBase):
    def setup(self, bkpt_str):
        self.build()
        
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, bkpt_str, lldb.SBFileSpec('main.c'))
        return thread

    def test_direct_call(self):
        thread = self.setup('Break here for direct')

        # Sanity check that we start out in the correct function.
        name = thread.frames[0].GetFunctionName()
        self.assertIn('direct_trampoline_call', name)

        # Check that stepping in will take us directly to the trampoline target.
        thread.StepInto()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('foo', name)

        # Check that stepping out takes us back to the trampoline caller.
        thread.StepOut()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('direct_trampoline_call', name)

        # Check that stepping over the end of the trampoline target 
        # takes us back to the trampoline caller.
        thread.StepInto()
        thread.StepOver()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('direct_trampoline_call', name)


    def test_chained_call(self):
        thread = self.setup('Break here for chained')

        # Sanity check that we start out in the correct function.
        name = thread.frames[0].GetFunctionName()
        self.assertIn('chained_trampoline_call', name)

        # Check that stepping in will take us directly to the trampoline target.
        thread.StepInto()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('foo', name)

        # Check that stepping out takes us back to the trampoline caller.
        thread.StepOut()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('chained_trampoline_call', name)

        # Check that stepping over the end of the trampoline target 
        # takes us back to the trampoline caller.
        thread.StepInto()
        thread.StepOver()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('chained_trampoline_call', name)

    def test_trampoline_after_nodebug(self):
        thread = self.setup('Break here for nodebug then trampoline')

        # Sanity check that we start out in the correct function.
        name = thread.frames[0].GetFunctionName()
        self.assertIn('trampoline_after_nodebug', name)

        # Check that stepping in will take us directly to the trampoline target.
        thread.StepInto()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('foo', name)

        # Check that stepping out takes us back to the trampoline caller.
        thread.StepOut()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('trampoline_after_nodebug', name)

        # Check that stepping over the end of the trampoline target 
        # takes us back to the trampoline caller.
        thread.StepInto()
        thread.StepOver()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('trampoline_after_nodebug', name)

    def test_unused_target(self):
        thread = self.setup('Break here for unused')

        # Sanity check that we start out in the correct function.
        name = thread.frames[0].GetFunctionName()
        self.assertIn('unused_target', name)

        # Check that stepping into a trampoline that doesn't call its target
        # jumps back to its caller.
        thread.StepInto()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('unused_target', name)
        
