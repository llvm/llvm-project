"""
Test lldb-dap stack trace response
"""


import dap_server
from lldbsuite.test.decorators import *

import lldbdap_testcase
from lldbsuite.test.lldbtest import *


class TestDAP_subtleFrames(lldbdap_testcase.DAPTestCaseBase):
    @add_test_categories(["libc++"])
    def test_subtleFrames(self):
        """
        Internal stack frames (such as the ones used by `std::function`) are marked as "subtle".
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        self.set_source_breakpoints(source, [line_number(source, "BREAK HERE")])
        self.continue_to_next_stop()

        frames = self.get_stackFrames()
        for f in frames:
            if "__function" in f["name"]:
                self.assertEqual(f["presentationHint"], "subtle")
        self.assertTrue(any(f.get("presentationHint") == "subtle" for f in frames))
