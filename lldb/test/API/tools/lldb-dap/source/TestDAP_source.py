"""
Test lldb-dap source request
"""


import os

import lldbdap_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestDAP_source(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    def test_stackTrace(self):
        """
        Tests the 'source' packet.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.c"
        self.source_path = os.path.join(os.getcwd(), source)
        self.qsort_call = line_number(source, "qsort call")

        lines = [self.qsort_call]
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )

        self.continue_to_breakpoints(breakpoint_ids)

        response = self.dap_server.request_source(sourceReference=0)
        self.assertFalse(response["success"], "verify invalid sourceReference fails")

        (stackFrames, totalFrames) = self.get_stackFrames_and_totalFramesCount()
        frameCount = len(stackFrames)
        self.assertGreaterEqual(
            frameCount, 3, "verify we get frames from system librarys (libc qsort)"
        )
        self.assertEqual(
            totalFrames,
            frameCount,
            "verify total frames returns a speculative page size",
        )
        expectedFrames = [
            {
                "name": "comp",
                "line": 14,
                "sourceName": "main.c",
                "containsSourceReference": False,
            },
            {"name": "qsort", "sourceName": "qsort", "containsSourceReference": True},
            {
                "name": "main",
                "line": 25,
                "sourceName": "main.c",
                "containsSourceReference": False,
            },
        ]
        for idx, expected in enumerate(expectedFrames):
            frame = stackFrames[idx]
            frame_name = self.get_dict_value(frame, ["name"])
            self.assertRegex(frame_name, expected["name"])
            source_name = self.get_dict_value(frame, ["source", "name"])
            self.assertRegex(source_name, expected["sourceName"])
            if expected["containsSourceReference"]:
                sourceReference = self.get_dict_value(
                    frame, ["source", "sourceReference"]
                )
                response = self.dap_server.request_source(
                    sourceReference=sourceReference
                )
                self.assertTrue(response["success"])
                self.assertGreater(
                    len(self.get_dict_value(response, ["body", "content"])),
                    0,
                    "verify content returned",
                )
                self.assertEqual(
                    self.get_dict_value(response, ["body", "mimeType"]),
                    "text/x-lldb.disassembly",
                    "verify mime type returned",
                )
            else:
                self.assertNotIn("sourceReference", frame["source"])
