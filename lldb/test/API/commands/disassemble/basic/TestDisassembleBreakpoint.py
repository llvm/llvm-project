"""
Test some lldb command abbreviations.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DisassemblyTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()
        target, _, _, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )
        self.runCmd("dis -f")
        disassembly_with_break = self.res.GetOutput().splitlines()

        self.assertTrue(target.BreakpointDelete(bkpt.GetID()))

        self.runCmd("dis -f")
        disassembly_without_break = self.res.GetOutput().splitlines()

        # Make sure all assembly instructions are the same as instructions
        # with the breakpoint removed.
        self.assertEqual(len(disassembly_with_break), len(disassembly_without_break))
        for dis_inst_with, dis_inst_without in zip(
            disassembly_with_break, disassembly_without_break
        ):
            inst_with = dis_inst_with.split(":")[-1]
            inst_without = dis_inst_without.split(":")[-1]
            self.assertEqual(inst_with, inst_without)
