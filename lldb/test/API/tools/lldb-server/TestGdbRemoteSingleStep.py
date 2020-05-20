

import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemoteSingleStep(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet
    @debugserver_test
    def test_single_step_only_steps_one_instruction_with_s_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.single_step_only_steps_one_instruction(
            use_Hc_packet=True, step_instruction="s")

    @skipIfWindows # No pty support to test any inferior std -i/e/o
    @llgs_test
    @skipIf(triple='^mips')
    def test_single_step_only_steps_one_instruction_with_s_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.single_step_only_steps_one_instruction(
            use_Hc_packet=True, step_instruction="s")
