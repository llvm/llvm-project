"""
Use lldb Python API to disassemble raw machine code bytes
"""

import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DisassembleRawDataTestCase(TestBase):
    @no_debug_info_test
    @skipIfRemote
    def test_disassemble_raw_data(self):
        """Test disassembling raw bytes with the API."""
        # Create a target from the debugger.
        arch = self.getArchitecture()
        if re.match("mips*el", arch):
            target = self.dbg.CreateTargetWithFileAndTargetTriple("", "mipsel")
            raw_bytes = bytearray([0x21, 0xF0, 0xA0, 0x03])
        elif re.match("mips", arch):
            target = self.dbg.CreateTargetWithFileAndTargetTriple("", "mips")
            raw_bytes = bytearray([0x03, 0xA0, 0xF0, 0x21])
        elif re.match("powerpc64le", arch):
            target = self.dbg.CreateTargetWithFileAndTargetTriple("", "powerpc64le")
            raw_bytes = bytearray([0x00, 0x00, 0x80, 0x38])
        elif arch in ("aarch64", "arm64"):
            target = self.dbg.CreateTargetWithFileAndTargetTriple("", "aarch64")
            raw_bytes = bytearray([0x60, 0x0C, 0x80, 0x52])
        elif arch == "arm":
            target = self.dbg.CreateTargetWithFileAndTargetTriple("", "arm")
            raw_bytes = bytearray([0x63, 0x30, 0xA0, 0xE3])
        else:
            target = self.dbg.CreateTargetWithFileAndTargetTriple("", "x86_64")
            raw_bytes = bytearray([0x48, 0x89, 0xE5])

        self.assertTrue(target, VALID_TARGET)
        insts = target.GetInstructions(lldb.SBAddress(0, target), raw_bytes)

        inst = insts.GetInstructionAtIndex(0)

        if self.TraceOn():
            print()
            print("Raw bytes:    ", [hex(x) for x in raw_bytes])
            print("Disassembled%s" % str(inst))
        if re.match("mips", arch):
            self.assertEqual(inst.GetMnemonic(target), "move")
            self.assertEqual(inst.GetOperands(target), "$" + "fp, " + "$" + "sp")
            self.assertEqual(
                inst.GetControlFlowKind(target), lldb.eInstructionControlFlowKindUnknown
            )
        elif re.match("powerpc64le", arch):
            self.assertEqual(inst.GetMnemonic(target), "li")
            self.assertEqual(inst.GetOperands(target), "4, 0")
            self.assertEqual(
                inst.GetControlFlowKind(target), lldb.eInstructionControlFlowKindUnknown
            )
        elif arch in ("aarch64", "arm64"):
            self.assertEqual(inst.GetMnemonic(target), "mov")
            self.assertEqual(inst.GetOperands(target), "w0, #0x63")
            self.assertEqual(
                inst.GetControlFlowKind(target), lldb.eInstructionControlFlowKindUnknown
            )
        elif arch == "arm":
            self.assertEqual(inst.GetMnemonic(target), "mov")
            self.assertEqual(inst.GetOperands(target), "r3, #99")
            self.assertEqual(
                inst.GetControlFlowKind(target), lldb.eInstructionControlFlowKindUnknown
            )
        else:
            self.assertEqual(inst.GetMnemonic(target), "movq")
            self.assertEqual(inst.GetOperands(target), "%" + "rsp, " + "%" + "rbp")
            self.assertEqual(
                inst.GetControlFlowKind(target), lldb.eInstructionControlFlowKindOther
            )
