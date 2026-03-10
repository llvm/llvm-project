"""
Test software step-inst, also known as instruction level single step, in risc-v atomic sequence.
For more information about atomic sequences, see the RISC-V Unprivileged ISA specification.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestSoftwareStep(TestBase):
    def do_sequence_test(self, filename, bkpt_name):
        source_file = filename + ".c"
        exe_file = filename + ".x"

        self.build(dictionary={"C_SOURCES": source_file, "EXE": exe_file})
        (target, process, cur_thread, bkpt) = lldbutil.run_to_name_breakpoint(
            self, bkpt_name, exe_name=exe_file
        )
        entry_pc = cur_thread.GetFrameAtIndex(0).GetPC()

        self.runCmd("thread step-inst")
        self.expect(
            "thread list",
            substrs=["stopped", "stop reason = instruction step into"],
        )

        # Get the instruction we stopped at
        pc = cur_thread.GetFrameAtIndex(0).GetPCAddress()
        inst = target.ReadInstructions(pc, 1).GetInstructionAtIndex(0)

        inst_mnemonic = inst.GetMnemonic(target)
        inst_operands = inst.GetOperands(target)
        if not inst_operands:
            return inst_mnemonic

        return f"{inst_mnemonic} {inst_operands}"

    @skipIf(archs=no_match("^riscv.*"))
    def test_cas(self):
        """
        This test verifies LLDB instruction step handling of a proper lr/sc pair.
        """
        instruction = self.do_sequence_test("main", "cas")
        self.assertEqual(instruction, "nop")

    @skipIf(archs=no_match("^riscv.*"))
    def test_branch_cas(self):
        """
        LLDB cannot predict the actual state of registers within a critical section (i.e., inside an atomic
        sequence). Therefore, it should identify all forward branches inside the atomic sequence and set
        breakpoints at every jump address that lies beyond the end of the sequence (after the sc instruction).
        This ensures that if any such branch is taken, execution will pause at its target address.

        This test includes an lr/sc sequence containing an active forward branch with a jump address located
        after the end of the atomic section. LLDB should correctly stop at this branch's target address. The
        test is nearly identical to the previous one, except for the branch condition, which is inverted and
        will result in a taken jump.
        """
        instruction = self.do_sequence_test("branch", "branch_cas")
        self.assertEqual(instruction, "ret")

    @skipIf(archs=no_match("^riscv.*"))
    def test_incomplete_sequence_without_lr(self):
        """
        This test verifies the behavior of a standalone sc instruction without a preceding lr. Since the sc
        lacks the required lr pairing, LLDB should treat it as a non-atomic store rather than part of an
        atomic sequence.
        """
        instruction = self.do_sequence_test(
            "incomplete_sequence_without_lr", "incomplete_cas"
        )
        self.assertEqual(instruction, "and a5, a2, a4")

    @skipIf(archs=no_match("^riscv.*"))
    def test_incomplete_sequence_without_sc(self):
        """
        This test checks the behavior of a standalone lr instruction without a subsequent sc. Since the lr
        lacks its required sc counterpart, LLDB should treat it as a non-atomic load rather than part of an
        atomic sequence.
        """
        instruction = self.do_sequence_test(
            "incomplete_sequence_without_sc", "incomplete_cas"
        )
        self.assertEqual(instruction, "and a5, a2, a4")
