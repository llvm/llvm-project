# lldb test suite imports
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import TestBase

# gdb-remote-specific imports
import lldbgdbserverutils
from gdbremote_testcase import GdbRemoteTestCaseBase


class TestGdbRemoteExitCode(GdbRemoteTestCaseBase):
    def _test_inferior_exit(self, retval):
        self.build()

        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["retval:%d" % retval]
        )

        self.test_sequence.add_log_lines(
            ["read packet: $vCont;c#a8", "send packet: $W{0:02x}#00".format(retval)],
            True,
        )

        self.expect_gdbremote_sequence()

    def test_inferior_exit_0(self):
        self._test_inferior_exit(0)

    def test_inferior_exit_42(self):
        self._test_inferior_exit(42)
