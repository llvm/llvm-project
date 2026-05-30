"""
Tests for the lldb-server mock accelerator plugin.

Verifies the accelerator-plugins+ feature in qSupported and
the jAcceleratorPluginInitialize packet response.
"""

import json

import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import configuration


def get_accelerator_action(actions, plugin_name):
    for action in actions:
        if action["plugin_name"] == plugin_name:
            return action
    return None


class MockAcceleratorPluginTestCase(gdbremote_testcase.GdbRemoteTestCaseBase):
    def setUp(self):
        super().setUp()
        if "mock-accelerator" not in configuration.enabled_plugins:
            self.skipTest("mock-accelerator plugin is not enabled")

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
    def test_jAcceleratorPluginInitialize_returns_mock_actions(self):
        self.build()
        self.set_inferior_startup_launch()
        self.prep_debug_monitor_and_inferior()

        self.add_qSupported_packets()
        self.expect_gdbremote_sequence()

        self.test_sequence.add_log_lines(
            [
                "read packet: $jAcceleratorPluginInitialize#00",
                {
                    "direction": "send",
                    "regex": r"^\$(.+)#[0-9a-fA-F]{2}",
                    "capture": {1: "accel_init_response"},
                },
            ],
            True,
        )
        context = self.expect_gdbremote_sequence()

        raw_response = context.get("accel_init_response")
        self.assertIsNotNone(raw_response)

        decoded = self.decode_gdbremote_binary(raw_response)
        actions = json.loads(decoded)
        self.assertIsInstance(actions, list)

        mock_action = get_accelerator_action(actions, "mock")
        self.assertIsNotNone(mock_action)
        self.assertIn("session_name", mock_action)
        self.assertIn("identifier", mock_action)
