"""
Test to ensure SBFrame::Disassemble produces SOME output
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class FrameDisassembleTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_disassemble(self):
        """Sample test to ensure SBFrame::Disassemble produces SOME output."""
        self.build()
        self.frame_disassemble_test()

    def frame_disassemble_test(self):
        """Sample test to ensure SBFrame::Disassemble produces SOME output"""
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        frame = thread.GetFrameAtIndex(0)
        disassembly = frame.Disassemble()
        self.assertNotEqual(disassembly, "")
        self.assertNotIn("error", disassembly)
        self.assertIn(": nop", disassembly)
