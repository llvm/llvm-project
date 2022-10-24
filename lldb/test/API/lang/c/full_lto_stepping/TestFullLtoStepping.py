"""Test that stepping in object files with multiple compile units works."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestFullLtoStepping(TestBase):

    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler="clang", compiler_version=['<', '13.0'])
    @skipUnlessDarwin
    def test(self):
        self.build()
        target = self.createTestTarget()
    
        breakpoint = target.BreakpointCreateByName("main")
        self.assertTrue(
            breakpoint and breakpoint.IsValid(),
            "Breakpoint is valid")

        _, _, thread, _ = lldbutil.run_to_breakpoint_do_run(self, target, breakpoint)
        name = thread.frames[0].GetFunctionName()
        # Check that we start out in main.
        self.assertEqual(name, 'main')
        thread.StepInto()
        name = thread.frames[0].GetFunctionName()
        # Check that we stepped into f.
        self.assertEqual(name, 'f')
