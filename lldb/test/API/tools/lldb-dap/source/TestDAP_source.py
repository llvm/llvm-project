"""
Test lldb-dap source request
"""


import os

import lldbdap_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestDAP_source(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    def test_source(self):
        """
        Tests the 'source' packet.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = self.getSourcePath("main.c")
        breakpoint_line = line_number(source, "breakpoint")

        lines = [breakpoint_line]
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )

        self.continue_to_breakpoints(breakpoint_ids)

        response = self.dap_server.request_source(sourceReference=0)
        self.assertFalse(response["success"], "verify invalid sourceReference fails")

        (stackFrames, totalFrames) = self.get_stackFrames_and_totalFramesCount()
        frameCount = len(stackFrames)
        self.assertGreaterEqual(frameCount, 3, "verify we got up to main at least")
        self.assertEqual(
            totalFrames,
            frameCount,
            "verify total frames returns a speculative page size",
        )
        wantFrames = [
            {
                "name": "handler",
                "line": 8,
                "source": {
                    "name": "main.c",
                    "path": source,
                    "containsSourceReference": False,
                },
            },
            {
                "name": "add",
                "source": {
                    "name": "add",
                    "path": program + "`add",
                    "containsSourceReference": True,
                },
            },
            {
                "name": "main",
                "line": 12,
                "source": {
                    "name": "main.c",
                    "path": source,
                    "containsSourceReference": False,
                },
            },
        ]
        for idx, want in enumerate(wantFrames):
            got = stackFrames[idx]
            name = self.get_dict_value(got, ["name"])
            self.assertEqual(name, want["name"])

            if "line" in want:
                line = self.get_dict_value(got, ["line"])
                self.assertEqual(line, want["line"])

            wantSource = want["source"]
            source_name = self.get_dict_value(got, ["source", "name"])
            self.assertEqual(source_name, wantSource["name"])

            source_path = self.get_dict_value(got, ["source", "path"])
            self.assertEqual(source_path, wantSource["path"])

            if wantSource["containsSourceReference"]:
                sourceReference = self.get_dict_value(
                    got, ["source", "sourceReference"]
                )
                response = self.dap_server.request_source(
                    sourceReference=sourceReference
                )
                self.assertTrue(response["success"])
                self.assertGreater(
                    len(self.get_dict_value(response, ["body", "content"])),
                    0,
                    "verify content returned disassembly",
                )
                self.assertEqual(
                    self.get_dict_value(response, ["body", "mimeType"]),
                    "text/x-lldb.disassembly",
                    "verify mime type returned",
                )
            else:
                self.assertNotIn("sourceReference", got["source"])
