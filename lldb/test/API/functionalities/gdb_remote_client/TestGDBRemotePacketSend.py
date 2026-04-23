import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase
from lldbsuite.test import lldbutil


class TestGDBRemotePacketSend(GDBRemoteTestBase):
    class MyResponder(MockGDBServerResponder):
        def __init__(self):
            super().__init__()
            self.custom_packets_received = []

        def qfThreadInfo(self):
            return "m1"

        def qsThreadInfo(self):
            return "l"

        def qC(self):
            return "QC1"

        def haltReason(self):
            return "T05thread:1;"

        def other(self, packet):
            if packet.startswith("qTestPacket"):
                self.custom_packets_received.append(packet)
                return "response-for-" + packet
            return ""

    def test_packet_send_multiple_arguments(self):
        """Each packet argument should be forwarded individually."""
        responder = self.MyResponder()
        self.server.responder = responder

        target = self.createTarget("a.yaml")
        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        self.expect(
            "process plugin packet send qTestPacketA qTestPacketB qTestPacketC",
            substrs=[
                "packet: qTestPacketA",
                "response: response-for-qTestPacketA",
                "packet: qTestPacketB",
                "response: response-for-qTestPacketB",
                "packet: qTestPacketC",
                "response: response-for-qTestPacketC",
            ],
        )

        self.assertEqual(
            responder.custom_packets_received,
            ["qTestPacketA", "qTestPacketB", "qTestPacketC"],
        )
