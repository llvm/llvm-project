"""Test that inlined argument variables have their correct location in debuginfo"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestRedefinitionsInInlines(TestBase):
    # https://github.com/llvm/llvm-project/issues/28219
    @skipIf(compiler="clang", compiler_version=["<", "3.5"])
    def test(self):
        self.source = "main.c"
        self.build()
        (target, process, thread, bp1) = lldbutil.run_to_source_breakpoint(
            self, "first breakpoint", lldb.SBFileSpec(self.source, False)
        )

        bp2 = target.BreakpointCreateBySourceRegex(
            "second breakpoint", lldb.SBFileSpec(self.source, False)
        )
        bp3 = target.BreakpointCreateBySourceRegex(
            "third breakpoint", lldb.SBFileSpec(self.source, False)
        )

        # When called from main(), test2 is passed in the value of 42 in 'b'
        self.expect("expression b", DATA_TYPES_DISPLAYED_CORRECTLY, substrs=["42"])

        process.Continue()

        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)
        bp_id = thread.GetStopReasonDataAtIndex(0)
        self.assertEqual(bp_id, bp2.GetID())

        self.expect("expression b", DATA_TYPES_DISPLAYED_CORRECTLY, substrs=["42"])
        self.expect("expression c", DATA_TYPES_DISPLAYED_CORRECTLY, substrs=["84"])

        process.Continue()

        # Now we're in test1(), and the first thing it does is call test2(24).  "Step in"
        # and check that we have the value 24 as the argument.
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)
        bp_id = thread.GetStopReasonDataAtIndex(0)
        self.assertEqual(bp_id, bp3.GetID())

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsInlined())
        self.assertEqual(frame.GetFunctionName(), "test1")

        thread.StepInto()

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsInlined())
        self.assertEqual(frame.GetFunctionName(), "test2")

        self.expect("expression b", DATA_TYPES_DISPLAYED_CORRECTLY, substrs=["24"])
