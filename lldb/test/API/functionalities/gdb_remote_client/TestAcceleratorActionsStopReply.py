import json

from lldbsuite.test.gdbclientutils import MockGDBServerResponder, escape_binary
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


def hex_json(value):
    return json.dumps(value, separators=(",", ":")).encode().hex()


class TestAcceleratorActionsStopReply(GDBRemoteTestBase):
    def test_actions_are_parsed_and_deduplicated_from_stop_reply(self):
        actions = []
        for plugin_name, identifier, address in (
            ("mock", 1, 0x1000),
            ("mock", 2, 0x2000),
            ("mock", 1, 0x1000),
        ):
            actions.append(
                {
                    "plugin_name": plugin_name,
                    "session_name": "",
                    "identifier": identifier,
                    "breakpoints": [
                        {
                            "identifier": identifier,
                            "by_address": {"load_address": address},
                            "symbol_names": [],
                        }
                    ],
                }
            )

        class Responder(MockGDBServerResponder):
            def __init__(self):
                super().__init__()
                self.breakpoint_packets = []
                self.clear_breakpoint_packets = []

            def qSupported(self, client_supported):
                return "PacketSize=3fff;QStartNoAckMode+;accelerator-plugins+"

            def haltReason(self):
                fields = "".join(
                    "accelerator-actions:%s;" % hex_json(action) for action in actions
                )
                return "T05thread:1;%s" % fields

            def setBreakpoint(self, packet):
                self.breakpoint_packets.append(packet)
                return "OK"

            def clearBreakpoint(self, packet):
                self.clear_breakpoint_packets.append(packet)
                return "OK"

            def other(self, packet):
                if packet == "jAcceleratorPluginInitialize":
                    return escape_binary("[]")
                return ""

        self.server.responder = Responder()
        target = self.dbg.CreateTargetWithFileAndTargetTriple("", "x86_64-pc-linux")
        process = self.connect(target)
        self.assertTrue(process.IsValid())

        self.assertEqual(len(self.server.responder.breakpoint_packets), 2)
        self.assertTrue(
            any(
                packet.startswith("Z0,1000,")
                for packet in self.server.responder.breakpoint_packets
            )
        )
        self.assertTrue(
            any(
                packet.startswith("Z0,2000,")
                for packet in self.server.responder.breakpoint_packets
            )
        )
