"""
Test software step-inst
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestSoftwareStep(TestBase):
    @skipIf(archs=no_match(re.compile("rv*")))
    def test_cas(self):
        self.build()
        (target, process, cur_thread, bkpt) = lldbutil.run_to_name_breakpoint(
            self, "cas"
        )
        entry_pc = cur_thread.GetFrameAtIndex(0).GetPC()

        self.runCmd("thread step-inst")
        self.expect(
            "thread list",
            substrs=["stopped", "stop reason = instruction step into"],
        )

        pc = cur_thread.GetFrameAtIndex(0).GetPC()
        self.assertTrue((pc - entry_pc) > 0x10)

    @skipIf(archs=no_match(re.compile("rv*")))
    def test_branch_cas(self):
        self.build(dictionary={"C_SOURCES": "branch.c", "EXE": "branch.x"})
        (target, process, cur_thread, bkpt) = lldbutil.run_to_name_breakpoint(
            self, "branch_cas", exe_name="branch.x"
        )
        entry_pc = cur_thread.GetFrameAtIndex(0).GetPC()

        self.runCmd("thread step-inst")
        self.expect(
            "thread list",
            substrs=["stopped", "stop reason = instruction step into"],
        )

        pc = cur_thread.GetFrameAtIndex(0).GetPC()
        self.assertTrue((pc - entry_pc) > 0x10)

    @skipIf(archs=no_match(re.compile("rv*")))
    def test_incomplete_sequence_without_lr(self):
        self.build(
            dictionary={
                "C_SOURCES": "incomplete_sequence_without_lr.c",
                "EXE": "incomplete_lr.x",
            }
        )
        (target, process, cur_thread, bkpt) = lldbutil.run_to_name_breakpoint(
            self, "incomplete_cas", exe_name="incomplete_lr.x"
        )
        entry_pc = cur_thread.GetFrameAtIndex(0).GetPC()

        self.runCmd("thread step-inst")

        self.expect(
            "thread list",
            substrs=["stopped", "stop reason = instruction step into"],
        )

        pc = cur_thread.GetFrameAtIndex(0).GetPC()
        self.assertTrue((pc - entry_pc) == 0x4)

    @skipIf(archs=no_match(re.compile("rv*")))
    def test_incomplete_sequence_without_sc(self):
        self.build(
            dictionary={
                "C_SOURCES": "incomplete_sequence_without_sc.c",
                "EXE": "incomplete_sc.x",
            }
        )
        (target, process, cur_thread, bkpt) = lldbutil.run_to_name_breakpoint(
            self, "incomplete_cas", exe_name="incomplete_sc.x"
        )
        entry_pc = cur_thread.GetFrameAtIndex(0).GetPC()

        self.runCmd("thread step-inst")

        self.expect(
            "thread list",
            substrs=["stopped", "stop reason = instruction step into"],
        )

        pc = cur_thread.GetFrameAtIndex(0).GetPC()
        self.assertTrue((pc - entry_pc) == 0x4)
