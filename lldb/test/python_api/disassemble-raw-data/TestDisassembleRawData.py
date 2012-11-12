"""
Use lldb Python API to disassemble raw machine code bytes
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class DisassembleRawDataTestCase(TestBase):

    mydir = os.path.join("python_api", "disassemble-raw-data")

    @python_api_test
    def test_disassemble_raw_data(self):
        """Test disassembling raw bytes with the API."""
        self.disassemble_raw_data()

    def disassemble_raw_data(self):
        """Test disassembling raw bytes with the API."""
        # Create a target from the debugger.

        target = self.dbg.CreateTargetWithFileAndTargetTriple ("", "x86_64")
        self.assertTrue(target, VALID_TARGET)

        raw_bytes = bytearray([0x48, 0x89, 0xe5])

        insts = target.GetInstructions(lldb.SBAddress(), raw_bytes)

        inst = insts.GetInstructionAtIndex(0)

        if self.TraceOn():
            print
            print "Raw bytes:    ", [hex(x) for x in raw_bytes]
            print "Disassembled%s" % str(inst)
 
        self.assertTrue (inst.GetMnemonic(target) == "movq")
        self.assertTrue (inst.GetOperands(target) == '%' + "rsp, " + '%' + "rbp")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
