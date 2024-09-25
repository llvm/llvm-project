"""
Test lldb-dap locations request
"""


import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import os


class TestDAP_locations(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    def test_locations(self):
        """
        Tests the 'locations' request.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.c"
        self.source_path = os.path.join(os.getcwd(), source)
        self.set_source_breakpoints(
            source,
            [line_number(source, "// BREAK HERE")],
        )
        self.continue_to_next_stop()

        locals = {l["name"]: l for l in self.dap_server.get_local_variables()}

        # var1 has a declarationLocation but no valueLocation
        self.assertIn("declarationLocationReference", locals["var1"].keys())
        self.assertNotIn("valueLocationReference", locals["var1"].keys())
        loc_var1 = self.dap_server.request_locations(
            locals["var1"]["declarationLocationReference"]
        )
        self.assertTrue(loc_var1["success"])
        self.assertTrue(loc_var1["body"]["source"]["path"].endswith("main.c"))
        self.assertEqual(loc_var1["body"]["line"], 2)
