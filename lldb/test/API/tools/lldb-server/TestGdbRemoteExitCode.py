
# lldb test suite imports
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import TestBase

# gdb-remote-specific imports
import lldbgdbserverutils
from gdbremote_testcase import GdbRemoteTestCaseBase


class TestGdbRemoteExitCode(GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def inferior_exit_0(self):
        self.prep_debug_monitor_and_inferior()
        self.test_sequence.add_log_lines(
            ["read packet: $vCont;c#a8",
             "send packet: $W00#00"],
            True)

        self.expect_gdbremote_sequence()

    @debugserver_test
    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet
    def test_inferior_exit_0_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.inferior_exit_0()

    @llgs_test
    def test_inferior_exit_0_llgs(self):
        self.init_llgs_test()
        self.build()
        self.inferior_exit_0()

    def inferior_exit_42(self):
        RETVAL = 42

        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["retval:%d" % RETVAL])

        self.test_sequence.add_log_lines(
            ["read packet: $vCont;c#a8",
             "send packet: $W{0:02x}#00".format(RETVAL)],
            True)

        self.expect_gdbremote_sequence()

    @debugserver_test
    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet
    def test_inferior_exit_42_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.inferior_exit_42()

    @llgs_test
    def test_inferior_exit_42_llgs(self):
        self.init_llgs_test()
        self.build()
        self.inferior_exit_42()
