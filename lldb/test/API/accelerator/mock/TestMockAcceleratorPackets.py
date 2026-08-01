"""
Packet-level tests for the lldb-server mock accelerator plugin.

Verifies the accelerator-plugins+ feature in qSupported,
the jAcceleratorPluginInitialize packet response, and
the jAcceleratorPluginBreakpointHit round-trip (including connect_info).
"""

import json

import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import escape_binary
from lldbsuite.test.lldbtest import *
from lldbsuite.test import configuration


def get_accelerator_action(actions, plugin_name):
    for action in actions:
        if action["plugin_name"] == plugin_name:
            return action
    return None


class MockAcceleratorPacketsTestCase(gdbremote_testcase.GdbRemoteTestCaseBase):
    def setUp(self):
        super().setUp()
        if "mock-accelerator" not in configuration.enabled_plugins:
            self.skipTest("mock-accelerator plugin is not enabled")

    def send_and_decode_json(self, packet):
        """Send a packet and return the decoded JSON response."""
        self.test_sequence.add_log_lines(
            [
                "read packet: $%s#00" % packet,
                {
                    "direction": "send",
                    "regex": r"^\$(.+)#[0-9a-fA-F]{2}",
                    "capture": {1: "response"},
                },
            ],
            True,
        )
        context = self.expect_gdbremote_sequence()
        raw = context.get("response")
        self.assertIsNotNone(raw)
        return json.loads(self.decode_gdbremote_binary(raw))

    @add_test_categories(["llgs"])
    def test_qSupported_reports_accelerator_plugins(self):
        self.build()
        self.set_inferior_startup_launch()
        self.prep_debug_monitor_and_inferior()

        self.add_qSupported_packets()
        context = self.expect_gdbremote_sequence()
        features = self.parse_qSupported_response(context)

        self.assertIn("accelerator-plugins", features)
        self.assertEqual(features["accelerator-plugins"], "+")

    @add_test_categories(["llgs"])
    def test_jAcceleratorPluginInitialize_returns_breakpoints(self):
        self.build()
        self.set_inferior_startup_launch()
        self.prep_debug_monitor_and_inferior()

        self.add_qSupported_packets()
        self.expect_gdbremote_sequence()

        actions = self.send_and_decode_json("jAcceleratorPluginInitialize")
        self.assertIsInstance(actions, list)

        mock_action = get_accelerator_action(actions, "mock")
        self.assertIsNotNone(mock_action)
        self.assertIn("breakpoints", mock_action)

        breakpoints = mock_action["breakpoints"]
        self.assertGreater(len(breakpoints), 0)

        bp = breakpoints[0]
        self.assertIn("identifier", bp)
        self.assertIn("by_name", bp)
        self.assertEqual(
            bp["by_name"]["function_name"], "mock_gpu_accelerator_initialize"
        )
        # The initialize breakpoint requests the "mock_gpu_accelerator_compute"
        # symbol value.
        self.assertEqual(bp["symbol_names"], ["mock_gpu_accelerator_compute"])

    @add_test_categories(["llgs"])
    def test_jAcceleratorPluginBreakpointHit_response(self):
        self.build()
        self.set_inferior_startup_launch()
        self.prep_debug_monitor_and_inferior()

        self.add_qSupported_packets()
        self.expect_gdbremote_sequence()

        # Simulate the initialize breakpoint being hit, supplying the requested
        # "mock_gpu_accelerator_compute" symbol value so the plugin sets a
        # breakpoint by address.
        hit_args = {
            "plugin_name": "mock",
            "breakpoint": {
                "identifier": 1,
                "symbol_names": ["mock_gpu_accelerator_compute"],
            },
            "symbol_values": [
                {"name": "mock_gpu_accelerator_compute", "value": 0x4000}
            ],
        }
        hit_json = json.dumps(hit_args, separators=(",", ":"))
        escaped_json = escape_binary(hit_json)
        response = self.send_and_decode_json(
            "jAcceleratorPluginBreakpointHit:" + escaped_json
        )

        self.assertIn("disable_bp", response)
        self.assertTrue(response["disable_bp"])
        self.assertIn("auto_resume_native", response)
        self.assertFalse(response["auto_resume_native"])

        # Verify the response includes new actions with both a by-name (scoped
        # to a shared library) and a by-address breakpoint.
        self.assertIn("actions", response)
        actions = response["actions"]
        self.assertEqual(actions["plugin_name"], "mock")
        self.assertIn("breakpoints", actions)
        new_bps = {bp["identifier"]: bp for bp in actions["breakpoints"]}

        # Breakpoint by name scoped to a shared library.
        self.assertIn(3, new_bps)
        self.assertEqual(
            new_bps[3]["by_name"]["function_name"], "mock_gpu_accelerator_finish"
        )
        self.assertEqual(new_bps[3]["by_name"]["shlib"], "a.out")

        # Breakpoint by address, using the supplied
        # "mock_gpu_accelerator_compute" symbol value.
        self.assertIn(2, new_bps)
        self.assertEqual(new_bps[2]["by_address"]["load_address"], 0x4000)

        # The initialize hit also arms the connection-trigger breakpoint by name.
        self.assertIn(4, new_bps)
        self.assertEqual(
            new_bps[4]["by_name"]["function_name"], "mock_gpu_accelerator_connect"
        )

    @add_test_categories(["llgs"])
    def test_jAcceleratorPluginBreakpointHit_returns_connect_info(self):
        self.build()
        self.set_inferior_startup_launch()
        self.prep_debug_monitor_and_inferior()

        self.add_qSupported_packets()
        self.expect_gdbremote_sequence()

        # Simulate the connection-trigger breakpoint (identifier 4) being hit.
        # The plugin responds with connect_info describing the connection the
        # client should make to the mock accelerator GDB server.
        hit_args = {
            "plugin_name": "mock",
            "breakpoint": {"identifier": 4, "symbol_names": []},
            "symbol_values": [],
        }
        hit_json = json.dumps(hit_args, separators=(",", ":"))
        escaped_json = escape_binary(hit_json)
        response = self.send_and_decode_json(
            "jAcceleratorPluginBreakpointHit:" + escaped_json
        )

        self.assertTrue(response["disable_bp"])
        self.assertFalse(response["auto_resume_native"])

        actions = response["actions"]
        self.assertEqual(actions["plugin_name"], "mock")
        self.assertEqual(actions["session_name"], "Mock Accelerator Session")

        # The connect_info must carry a connect URL to a local port and request
        # a synchronous connection.
        self.assertIn("connect_info", actions)
        connect_info = actions["connect_info"]
        self.assertTrue(
            connect_info["connect_url"].startswith("connect://localhost:"),
            connect_info["connect_url"],
        )
        self.assertTrue(connect_info["synchronous"])
