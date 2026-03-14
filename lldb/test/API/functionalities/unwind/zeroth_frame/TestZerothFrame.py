"""
Test that line information is recalculated properly for a frame when it moves
from the middle of the backtrace to a zero index.

This is a regression test for a StackFrame bug, where whether frame is zero or
not depends on an internal field. When LLDB was updating its frame list value
of the field wasn't copied into existing StackFrame instances, so those
StackFrame instances, would use an incorrect line entry evaluation logic in
situations if it was in the middle of the stack frame list (not zeroth), and
then moved to the top position. The difference in logic is that for zeroth
frames line entry is returned for program counter, while for other frame
(except for those that "behave like zeroth") it is for the instruction
preceding PC, as PC points to the next instruction after function call. When
the bug is present, when execution stops at the second breakpoint
SBFrame.GetLineEntry() returns line entry for the previous line, rather than
the one with a breakpoint. Note that this is specific to
SBFrame.GetLineEntry(), SBFrame.GetPCAddress().GetLineEntry() would return
correct entry.

This bug doesn't reproduce through an LLDB interpretator, however it happens
when using API directly, for example in LLDB-MI.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ZerothFrame(TestBase):
    def test(self):
        """
        Test that line information is recalculated properly for a frame when it moves
        from the middle of the backtrace to a zero index.
        """
        self.build()
        self.setTearDownCleanup()

        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        main_dot_c = lldb.SBFileSpec("main.c")
        bp1 = target.BreakpointCreateBySourceRegex(
            "// Set breakpoint 1 here", main_dot_c
        )
        bp2 = target.BreakpointCreateBySourceRegex(
            "// Set breakpoint 2 here", main_dot_c
        )

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, VALID_PROCESS)

        thread = self.thread()

        if self.TraceOn():
            print("Backtrace at the first breakpoint:")
            for f in thread.frames:
                print(f)

        # Check that we have stopped at correct breakpoint.
        self.assertEqual(
            thread.frame[0].GetLineEntry().GetLine(),
            bp1.GetLocationAtIndex(0).GetAddress().GetLineEntry().GetLine(),
            "LLDB reported incorrect line number.",
        )

        # Important to use SBProcess::Continue() instead of
        # self.runCmd('continue'), because the problem doesn't reproduce with
        # 'continue' command.
        process.Continue()

        if self.TraceOn():
            print("Backtrace at the second breakpoint:")
            for f in thread.frames:
                print(f)
        # Check that we have stopped at the breakpoint
        self.assertEqual(
            thread.frame[0].GetLineEntry().GetLine(),
            bp2.GetLocationAtIndex(0).GetAddress().GetLineEntry().GetLine(),
            "LLDB reported incorrect line number.",
        )
        # Double-check with GetPCAddress()
        self.assertEqual(
            thread.frame[0].GetLineEntry().GetLine(),
            thread.frame[0].GetPCAddress().GetLineEntry().GetLine(),
            "LLDB reported incorrect line number.",
        )
