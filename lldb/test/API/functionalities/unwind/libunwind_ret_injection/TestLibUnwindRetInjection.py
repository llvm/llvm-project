"""
Test that libunwind correctly injects 'ret' instructions to rebalance execution flow
when unwinding C++ exceptions. This is important for Apple Processor Trace analysis.
"""

import lldb
import os
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test import configuration


class LibunwindRetInjectionTestCase(TestBase):
    @skipIf(archs=no_match(["arm64", "arm64e", "aarch64"]))
    @skipUnlessDarwin
    @skipIfOutOfTreeLibunwind
    def test_ret_injection_on_exception_unwind(self):
        """Test that __libunwind_Registers_arm64_jumpto receives correct walkedFrames count and injects the right number of ret instructions."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Find the just-built libunwind, not the system one.
        # llvm_tools_dir is typically <build>/bin, so lib is a sibling.
        self.assertIsNotNone(
            configuration.llvm_tools_dir,
            "llvm_tools_dir must be set to find in-tree libunwind",
        )

        llvm_lib_dir = os.path.join(
            os.path.dirname(configuration.llvm_tools_dir), "lib"
        )

        # Find the libunwind library (platform-agnostic).
        libunwind_path = None
        for filename in os.listdir(llvm_lib_dir):
            if filename.startswith("libunwind.") or filename.startswith("unwind."):
                libunwind_path = os.path.join(llvm_lib_dir, filename)
                break

        self.assertIsNotNone(
            libunwind_path, f"Could not find libunwind in {llvm_lib_dir}"
        )

        # Set breakpoint in __libunwind_Registers_arm64_jumpto.
        # This is the function that performs the actual jump and ret injection.
        bp = target.BreakpointCreateByName("__libunwind_Registers_arm64_jumpto")
        self.assertTrue(bp.IsValid())
        self.assertGreater(bp.GetNumLocations(), 0)

        # Set up DYLD_INSERT_LIBRARIES to use the just-built libunwind.
        launch_info = lldb.SBLaunchInfo(None)
        env = target.GetEnvironment()
        env.Set("DYLD_INSERT_LIBRARIES", libunwind_path, True)
        launch_info.SetEnvironment(env, False)

        # Launch the process with our custom libunwind.
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertSuccess(
            error, f"Failed to launch process with libunwind at {libunwind_path}"
        )
        self.assertTrue(process, PROCESS_IS_VALID)

        # We should hit the breakpoint in __libunwind_Registers_arm64_jumpto
        # during the exception unwinding phase 2.
        threads = lldbutil.get_threads_stopped_at_breakpoint(process, bp)
        self.assertEqual(len(threads), 1, "Should have stopped at breakpoint")

        thread = threads[0]
        frame = thread.GetFrameAtIndex(0)

        # Verify we're in __libunwind_Registers_arm64_jumpto.
        function_name = frame.GetFunctionName()
        self.assertTrue(
            "__libunwind_Registers_arm64_jumpto" in function_name,
            f"Expected to be in __libunwind_Registers_arm64_jumpto, got {function_name}",
        )

        # On ARM64, the walkedFrames parameter should be in register x1 (second parameter).
        # According to the ARM64 calling convention, integer arguments are passed in x0-x7.
        # x0 = Registers_arm64* pointer.
        # x1 = unsigned walkedFrames.
        error = lldb.SBError()
        x1_value = frame.register["x1"].GetValueAsUnsigned(error)
        self.assertSuccess(error, "Failed to read x1 register")

        # According to the code in UnwindCursor.hpp, the walkedFrames value represents:
        # 1. The number of frames walked in unwind_phase2 to reach the landing pad.
        # 2. Plus _EXTRA_LIBUNWIND_FRAMES_WALKED = 5 - 1 = 4 additional libunwind frames.
        #
        # From the comment in the code:
        #   frame #0: __libunwind_Registers_arm64_jumpto
        #   frame #1: Registers_arm64::returnto
        #   frame #2: UnwindCursor::jumpto
        #   frame #3: __unw_resume
        #   frame #4: __unw_resume_with_frames_walked
        #   frame #5: unwind_phase2
        #
        # Since __libunwind_Registers_arm64_jumpto returns to the landing pad,
        # we subtract 1, so _EXTRA_LIBUNWIND_FRAMES_WALKED = 4.
        #
        # For our test program:
        # - unwind_phase2 starts walking (frame 0 counted here).
        # - Walks through: func_d (throw site), func_c, func_b, func_a.
        # - Finds landing pad in main.
        # That's approximately 4-5 frames from the user code.
        # Plus the 4 extra libunwind frames.
        #
        # So we expect x1 to be roughly 8-10.
        expected_min_frames = 8
        expected_max_frames = 13  # Allow some variation for libc++abi frames.

        self.assertGreaterEqual(
            x1_value,
            expected_min_frames,
            f"walkedFrames (x1) should be >= {expected_min_frames}, got {x1_value}. "
            "This is the number of 'ret' instructions that will be executed.",
        )

        self.assertLessEqual(
            x1_value,
            expected_max_frames,
            f"walkedFrames (x1) should be <= {expected_max_frames}, got {x1_value}. "
            "Value seems too high.",
        )

        # Now step through the ret injection loop and count the actual number of 'ret' executions.
        # The loop injects exactly x1_value ret instructions before continuing with register restoration.
        # We step until we hit the first 'ldp' instruction (register restoration starts with 'ldp x2, x3, [x0, #0x010]').
        ret_executed_count = 0
        max_steps = 100  # Safety limit to prevent infinite loops.

        for step_count in range(max_steps):
            # Get current instruction.
            pc = frame.GetPC()
            inst = process.ReadMemory(pc, 4, lldb.SBError())

            # Disassemble current instruction.
            current_inst = target.GetInstructions(lldb.SBAddress(pc, target), inst)[0]
            mnemonic = current_inst.GetMnemonic(target)
            operands = current_inst.GetOperands(target)

            # Check if we've reached the register restoration part (first ldp after the loop).
            if mnemonic == "ldp":
                # We've exited the ret injection loop.
                break

            # Count 'ret' instructions that get executed.
            if mnemonic == "ret":
                self.assertEqual(operands, "x16")
                ret_executed_count += 1

            # Step one instruction.
            thread.StepInstruction(False)  # False = step over.

            # Update frame reference.
            frame = thread.GetFrameAtIndex(0)

        # Verify we didn't hit the safety limit.
        self.assertLess(
            step_count,
            max_steps - 1,
            f"Stepped {max_steps} times without reaching 'ldp' instruction. Something is wrong.",
        )

        # The number of executed 'ret' instructions should match x1_value.
        # According to the implementation, the loop executes exactly x1_value times.
        self.assertEqual(
            ret_executed_count,
            x1_value,
            f"Expected {x1_value} 'ret' instructions to be executed (matching x1 register), "
            f"but counted {ret_executed_count} executed 'ret' instructions.",
        )
