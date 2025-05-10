"""Test that lldb backtraces a frameless function that faults correctly."""

import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import shutil
import os


class TestUnwindFramelessFaulted(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(
        oslist=no_match([lldbplatformutil.getDarwinOSTriples(), "linux"]),
        archs=no_match(["aarch64", "arm64", "arm64e"]),
    )
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
        frame = thread.GetFrameAtIndex(0)
        step_count = 0
        max_step_count = 200
        while (
            process.GetState() == lldb.eStateStopped
            and frame.name in correct_frames
            and step_count < max_step_count
        ):
            starting_index = 0
            if self.TraceOn():
                self.runCmd("bt")

            # Find which index into correct_frames the current stack frame is
            for idx, name in enumerate(correct_frames):
                if frame.name == name:
                    starting_index = idx

            # Test that all frames after the current frame listed in
            # correct_frames appears in the backtrace.
            frame_idx = 0
            for expected_frame in correct_frames[starting_index:]:
                self.assertEqual(thread.GetFrameAtIndex(frame_idx).name, expected_frame)
                frame_idx = frame_idx + 1

            # When we're at our deepest level, test that register passing of
            # x0 and x20 follow the by-hand UnwindPlan rules.
            # In this test program, we can get x0 in the middle of the stack
            # and we CAN'T get x20. The opposites of the normal AArch64 SysV
            # ABI.
            if frame.name == "break_to_debugger":
                tbi_frame = thread.GetFrameAtIndex(2)
                self.assertEqual(tbi_frame.name, "to_be_interrupted")
                # The original argument to to_be_interrupted(), 10
                # Normally can't get x0 mid-stack, but UnwindPlans have
                # special rules to make this possible.
                x0_reg = tbi_frame.register["x0"]
                self.assertTrue(x0_reg.IsValid())
                self.assertEqual(x0_reg.GetValueAsUnsigned(), 10)
                # The incremented return value from to_be_interrupted(), 11
                x24_reg = tbi_frame.register["x24"]
                self.assertTrue(x24_reg.IsValid())
                self.assertEqual(x24_reg.GetValueAsUnsigned(), 11)
                # x20 can normally be fetched mid-stack, but the UnwindPlan
                # has a rule saying it can't be fetched.
                x20_reg = tbi_frame.register["x20"]
                self.assertTrue(x20_reg.error.fail)

                trap_frame = thread.GetFrameAtIndex(1)
                self.assertEqual(trap_frame.name, "trap")
                # Confirm that we can fetch x0 from trap() which
                # is normally not possible w/ SysV AbI, but special
                # UnwindPlans in use.
                x0_reg = trap_frame.register["x0"]
                self.assertTrue(x0_reg.IsValid())
                self.assertEqual(x0_reg.GetValueAsUnsigned(), 10)
                x1_reg = trap_frame.register["x1"]
                self.assertTrue(x1_reg.error.fail)

                main_frame = thread.GetFrameAtIndex(3)
                self.assertEqual(main_frame.name, "main")
                # x20 can normally be fetched mid-stack, but the UnwindPlan
                # has a rule saying it can't be fetched.
                x20_reg = main_frame.register["x20"]
                self.assertTrue(x20_reg.error.fail)
                # x21 can be fetched mid-stack.
                x21_reg = main_frame.register["x21"]
                self.assertTrue(x21_reg.error.success)

            # manually move past the BRK instruction in
            # break_to_debugger().  lldb-server doesn't
            # advance past the builtin_debugtrap() BRK
            # instruction.
            if (
                thread.GetStopReason() == lldb.eStopReasonException
                and frame.name == "break_to_debugger"
            ):
                frame.SetPC(frame.GetPC() + 4)

            if self.TraceOn():
                print("StepInstruction")
            thread.StepInstruction(False)
            frame = thread.GetFrameAtIndex(0)
            step_count = step_count + 1
