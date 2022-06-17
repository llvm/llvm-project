"""
Use lldb Python API to disassemble raw machine code bytes
"""

from __future__ import print_function

from io import StringIO
import sys

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class Disassemble_VST1_64(TestBase):

    @no_debug_info_test
    @skipIfLLVMTargetMissing("ARM")
    def test_disassemble_invalid_vst_1_64_raw_data(self):
        """Test disassembling invalid vst1.64 raw bytes with the API."""
        # Create a target from the debugger.
        target = self.dbg.CreateTargetWithFileAndTargetTriple("", "thumbv7-apple-macosx")
        self.assertTrue(target, VALID_TARGET)

        raw_bytes = bytearray([0xf0, 0xb5, 0x03, 0xaf,
                               0x2d, 0xe9, 0x00, 0x0d,
                               0xad, 0xf1, 0x40, 0x04,
                               0x24, 0xf0, 0x0f, 0x04,
                               0xa5, 0x46])

        assembly = """
        push   {r4, r5, r6, r7, lr}
        add    r7, sp, #0xc
        push.w {r8, r10, r11}
        sub.w  r4, sp, #0x40
        bic    r4, r4, #0xf
        mov    sp, r4
        """
        def split(s):
            return [x.strip() for x in s.strip().splitlines()]

        insts = target.GetInstructions(lldb.SBAddress(), raw_bytes)

        if self.TraceOn():
            print()
            for i in insts:
                print("Disassembled %s" % str(i))

        sio = StringIO()
        insts.Print(sio)
        self.assertEqual(split(assembly), split(sio.getvalue()))

        self.assertEqual(insts.GetSize(), len(split(assembly)))

        for i,asm in enumerate(split(assembly)):
            inst = insts.GetInstructionAtIndex(i)
            sio = StringIO()
            inst.Print(sio)
            self.assertEqual(asm, sio.getvalue().strip())

        raw_bytes = bytearray([0x04, 0xf9, 0xed, 0x82])

        insts = target.GetInstructions(lldb.SBAddress(), raw_bytes)

        inst = insts.GetInstructionAtIndex(0)

        if self.TraceOn():
            print()
            print("Raw bytes:    ", [hex(x) for x in raw_bytes])
            print("Disassembled%s" % str(inst))

        self.assertEqual(inst.GetMnemonic(target), "vst1.64")
