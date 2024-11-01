""" Check that unavailable registers are reported when reading register sets."""

from textwrap import dedent
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class MyResponder(MockGDBServerResponder):
    def readRegisters(self):
        return "E01"

    def readRegister(self, regnum):
        if regnum in [0, 1, 2]:
            return "E01"
        return "5555555555555555"

    def qHostInfo(self):
        # The triple is hex encoded ASCII "x86_64-linux-gnu".
        return "triple:7838365F36342D6C696E75782D676E75;"


class TestRegistersUnavailable(GDBRemoteTestBase):
    @skipIfRemote
    # So that we have multiple register sets.
    @skipIfLLVMTargetMissing("X86")
    def test_unavailable_registers(self):
        self.server.responder = MyResponder()
        target = self.dbg.CreateTarget("")

        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets process")
            self.addTearDownHook(
                lambda: self.runCmd("log disable gdb-remote packets process")
            )

        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        # We are using a long regex pattern here to check that the indentation
        # is correct when you have multiple register sets and they all have
        # some missing registers.
        self.expect(
            "register read --all",
            patterns=[
                "(?sm)^general purpose registers:\n"
                "^\s+rdx = 0x5555555555555555\n"
                ".*"
                "^3 registers were unavailable.\n"
                "\n"
                "^supplementary registers:\n"
                "^\s+edx = 0x55555555\n"
                ".*"
                "^12 registers were unavailable."
            ],
        )
