import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemoteSingleStep(gdbremote_testcase.GdbRemoteTestCaseBase):
    @skipIfWindows  # No pty support to test any inferior std -i/e/o
    @skipIf(triple="^mips")
    def test_single_step_only_steps_one_instruction_with_s(self):
        self.build()
        self.set_inferior_startup_launch()
        self.single_step_only_steps_one_instruction(
            use_Hc_packet=True, step_instruction="s"
        )
