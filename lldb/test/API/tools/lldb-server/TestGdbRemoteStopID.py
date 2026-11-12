import re

import gdbremote_testcase
from lldbsuite.test.decorators import add_test_categories


class TestGdbRemoteStopID(gdbremote_testcase.GdbRemoteTestCaseBase):
    @add_test_categories(["llgs"])
    def test_stop_id_advances(self):
        self.build()
        self.set_inferior_startup_launch()
        self.prep_debug_monitor_and_inferior(inferior_args=["trap", "trap"])

        self.test_sequence.add_log_lines(
            [
                "read packet: $c#63",
                {
                    "direction": "send",
                    "regex": r"^\$T[0-9a-fA-F]{2}([^#]*)#[0-9a-fA-F]{2}$",
                    "capture": {1: "first_stop_reply"},
                },
                "read packet: $c#63",
                {
                    "direction": "send",
                    "regex": r"^\$T[0-9a-fA-F]{2}([^#]*)#[0-9a-fA-F]{2}$",
                    "capture": {1: "second_stop_reply"},
                },
            ],
            True,
        )

        context = self.expect_gdbremote_sequence()
        first = self.parse_key_val_dict(context["first_stop_reply"])["stop_id"]
        second = self.parse_key_val_dict(context["second_stop_reply"])["stop_id"]
        self.assertRegex(first, re.compile(r"^[0-9]+$"))
        self.assertRegex(second, re.compile(r"^[0-9]+$"))
        self.assertGreater(int(second), int(first))
