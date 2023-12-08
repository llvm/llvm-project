import json
import gdbremote_testcase
import lldbgdbserverutils
from lldbsuite.support import seven
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemoteModuleInfo(gdbremote_testcase.GdbRemoteTestCaseBase):
    @add_test_categories(["llgs"])
    def test_module_info(self):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior()
        self.add_process_info_collection_packets()
        context = self.expect_gdbremote_sequence()
        info = self.parse_process_info_response(context)

        self.test_sequence.add_log_lines(
            [
                "read packet: $jModulesInfo:%s]#00"
                % json.dumps(
                    [
                        {
                            "file": lldbutil.append_to_process_working_directory(
                                self, "a.out"
                            ),
                            "triple": seven.unhexlify(info["triple"]),
                        }
                    ]
                ),
                {
                    "direction": "send",
                    "regex": r"^\$\[{(.*)}\]\]#[0-9A-Fa-f]{2}",
                    "capture": {1: "spec"},
                },
            ],
            True,
        )

        context = self.expect_gdbremote_sequence()
        spec = context.get("spec")
        self.assertRegexpMatches(spec, '"file_path":".*"')
        self.assertRegexpMatches(spec, '"file_offset":\d+')
        self.assertRegexpMatches(spec, '"file_size":\d+')
        self.assertRegexpMatches(spec, '"triple":"\w*-\w*-.*"')
        self.assertRegexpMatches(spec, '"uuid":"[A-Fa-f0-9]+"')
