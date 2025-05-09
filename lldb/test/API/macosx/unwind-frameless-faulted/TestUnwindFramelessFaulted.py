"""Test that lldb backtraces a frameless function that faults correctly."""

import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import shutil
import os


class TestUnwindFramelessFaulted(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    def test_frameless_faulted_unwind(self):
        self.build()

        (target, process, thread, bp) = lldbutil.run_to_name_breakpoint(
            self, "main", only_one_thread=False
        )

        # The test program will have a backtrace like this at its deepest:
        #
        #   * frame #0: 0x0000000102adc468 a.out`break_to_debugger + 4
        #     frame #1: 0x0000000102adc458 a.out`trap + 16
        #     frame #2: 0x0000000102adc440 a.out`to_be_interrupted + 20
        #     frame #3: 0x0000000102adc418 a.out`main at main.c:4:7
        #     frame #4: 0x0000000193b7eb4c dyld`start + 6000

        correct_frames = ["break_to_debugger", "trap", "to_be_interrupted", "main"]

        # Keep track of when main has branch & linked, instruction step until we're
        # back in main()
        main_has_bl_ed = False

        # Instruction step through the binary until we are in a function not
        # listed in correct_frames.
        while (
            process.GetState() == lldb.eStateStopped
            and thread.GetFrameAtIndex(0).name in correct_frames
        ):
            starting_index = 0
            if self.TraceOn():
                self.runCmd("bt")

            # Find which index into correct_frames the current stack frame is
            for idx, name in enumerate(correct_frames):
                if thread.GetFrameAtIndex(0).name == name:
                    starting_index = idx

            # Test that all frames after the current frame listed in
            # correct_frames appears in the backtrace.
            frame_idx = 0
            for expected_frame in correct_frames[starting_index:]:
                self.assertEqual(thread.GetFrameAtIndex(frame_idx).name, expected_frame)
                frame_idx = frame_idx + 1

            if self.TraceOn():
                print("StepInstruction")
            thread.StepInstruction(False)
