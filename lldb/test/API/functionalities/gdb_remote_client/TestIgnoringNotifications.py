"""
Test that LLDB ignores notification packets (those beginning with '%')
when waiting for a response to a '$' packet.

OpenOCD is one debug server that uses notification packets to reset
the connection timeout if a memory read takes too long. So that's what
we test here, but it should apply to any exchange.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class NotifyingServerResponder(MockGDBServerResponder):
    def readMemory(self, addr, length):
        return "01" * length


class NotifyingServer(MockGDBServer):
    def __init__(self, socket):
        self._socket = socket
        self.responder = NotifyingServerResponder()
        self.sent_notification = False

    def _sendPacket(self, packet: str, prefix="$"):
        # In theory we could add notifies before all packets, but it always
        # goes through the same code and just makes the test take longer.
        if packet == "01010101":
            # Send more than one to make sure we clear them all to find
            # the real response.
            super()._sendPacket("this_is_a_notification", prefix="%")
            super()._sendPacket("this_is_a_2nd_notification", prefix="%")
            self.sent_notification = True

        super()._sendPacket(packet)

        if packet == "01010101":
            super()._sendPacket("this_is_a_3rd_notification", prefix="%")


class TestIgnoringNotifications(GDBRemoteTestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.server = NotifyingServer(self.server_socket_class())
        self.server.start()

    @skipIfLLVMTargetMissing("AArch64")
    def test(self):
        target = self.createTarget("basic_eh_frame-aarch64.yaml")

        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets")
            self.runCmd("log enable gdb-remote comm")
            self.addTearDownHook(lambda: self.runCmd("log disable gdb-remote packets"))

        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        # Disabling cache is not required but makes debugging this test easier.
        self.runCmd("settings set target.process.disable-memory-cache true")

        # Should succeed despite getting a notification before the real result.
        self.expect("memory read --format hex 0x1234 0x1238", substrs=["0x01010101"])

        # Check we didn't succeed because lldb got memory in some other way.
        self.assertTrue(self.server.sent_notification)
