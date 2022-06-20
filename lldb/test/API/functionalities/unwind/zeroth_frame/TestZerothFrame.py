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

from __future__ import print_function


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

        bp1_line = line_number('main.c', '// Set breakpoint 1 here')
        bp2_line = line_number('main.c', '// Set breakpoint 2 here')

        lldbutil.run_break_set_by_file_and_line(
            self,
            'main.c',
            bp1_line,
            num_expected_locations=1)
        lldbutil.run_break_set_by_file_and_line(
            self,
            'main.c',
            bp2_line,
            num_expected_locations=1)

        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, VALID_PROCESS)

        thread = process.GetThreadAtIndex(0)
        if self.TraceOn():
            print("Backtrace at the first breakpoint:")
            for f in thread.frames:
                print(f)
        # Check that we have stopped at correct breakpoint.
        self.assertEqual(
            process.GetThreadAtIndex(0).frame[0].GetLineEntry().GetLine(),
            bp1_line,
            "LLDB reported incorrect line number.")

        # Important to use SBProcess::Continue() instead of
        # self.runCmd('continue'), because the problem doesn't reproduce with
        # 'continue' command.
        process.Continue()

        thread = process.GetThreadAtIndex(0)
        if self.TraceOn():
            print("Backtrace at the second breakpoint:")
            for f in thread.frames:
                print(f)
        # Check that we have stopped at the breakpoint
        self.assertEqual(
            thread.frame[0].GetLineEntry().GetLine(),
            bp2_line,
            "LLDB reported incorrect line number.")
        # Double-check with GetPCAddress()
        self.assertEqual(
            thread.frame[0].GetLineEntry().GetLine(),
            thread.frame[0].GetPCAddress().GetLineEntry().GetLine(),
            "LLDB reported incorrect line number.")
