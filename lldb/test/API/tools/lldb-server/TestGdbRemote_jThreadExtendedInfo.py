import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import re
import json


@add_test_categories(["llgs"])
class TestGdbRemote_jThreadExtendedInfo(gdbremote_testcase.GdbRemoteTestCaseBase):
    def test(self):
        self.build()
        self.prep_debug_monitor_and_inferior()

        self.add_threadinfo_collection_packets()
        self.test_sequence.add_log_lines(
            [
                "read packet: $jThreadExtendedInfo:#00",
                "send packet: $OK#00",
            ],
            True,
        )
        context = self.expect_gdbremote_sequence()
        threads = self.parse_threadinfo_packets(context)
        self.assertEqual(len(threads), 1)
        self.test_sequence.add_log_lines(
            [
                f'read packet: $jThreadExtendedInfo:{{"thread":{threads[0]}}}]#00',
                {
                    "direction": "send",
                    "regex": re.compile(
                        r"^\$(.*)#[0-9a-fA-F]{2}$", re.MULTILINE | re.DOTALL
                    ),
                    "capture": {1: "content_raw"},
                },
            ],
            True,
        )
        context = self.expect_gdbremote_sequence()
        content_raw = context.get("content_raw")
        self.assertIsNotNone(content_raw)
        content = self.decode_gdbremote_binary(content_raw)
        triple = self.dbg.GetSelectedPlatform().GetTriple()
        # Only Windows has extended info.
        if re.match(".*-.*-windows", triple):
            dec = json.loads(content)
            self.assertIsInstance(dec, dict)
            self.assertIn("teb_address", dec)
            self.assertNotEqual(dec["teb_address"], 0)
        else:
            self.assertEqual(content, "")
