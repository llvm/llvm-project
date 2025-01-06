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
        source = "main.cpp"
        self.source_path = os.path.join(os.getcwd(), source)
        self.set_source_breakpoints(
            source,
            [line_number(source, "break here")],
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
        self.assertTrue(loc_var1["body"]["source"]["path"].endswith("main.cpp"))
        self.assertEqual(loc_var1["body"]["line"], 6)

        # func_ptr has both a declaration and a valueLocation
        self.assertIn("declarationLocationReference", locals["func_ptr"].keys())
        self.assertIn("valueLocationReference", locals["func_ptr"].keys())
        decl_loc_func_ptr = self.dap_server.request_locations(
            locals["func_ptr"]["declarationLocationReference"]
        )
        self.assertTrue(decl_loc_func_ptr["success"])
        self.assertTrue(
            decl_loc_func_ptr["body"]["source"]["path"].endswith("main.cpp")
        )
        self.assertEqual(decl_loc_func_ptr["body"]["line"], 7)
        val_loc_func_ptr = self.dap_server.request_locations(
            locals["func_ptr"]["valueLocationReference"]
        )
        self.assertTrue(val_loc_func_ptr["success"])
        self.assertTrue(val_loc_func_ptr["body"]["source"]["path"].endswith("main.cpp"))
        self.assertEqual(val_loc_func_ptr["body"]["line"], 3)

        # func_ref has both a declaration and a valueLocation
        self.assertIn("declarationLocationReference", locals["func_ref"].keys())
        self.assertIn("valueLocationReference", locals["func_ref"].keys())
        decl_loc_func_ref = self.dap_server.request_locations(
            locals["func_ref"]["declarationLocationReference"]
        )
        self.assertTrue(decl_loc_func_ref["success"])
        self.assertTrue(
            decl_loc_func_ref["body"]["source"]["path"].endswith("main.cpp")
        )
        self.assertEqual(decl_loc_func_ref["body"]["line"], 8)
        val_loc_func_ref = self.dap_server.request_locations(
            locals["func_ref"]["valueLocationReference"]
        )
        self.assertTrue(val_loc_func_ref["success"])
        self.assertTrue(val_loc_func_ref["body"]["source"]["path"].endswith("main.cpp"))
        self.assertEqual(val_loc_func_ref["body"]["line"], 3)

        # `evaluate` responses for function pointers also have locations associated
        eval_res = self.dap_server.request_evaluate("greet")
        self.assertTrue(eval_res["success"])
        self.assertIn("valueLocationReference", eval_res["body"].keys())
