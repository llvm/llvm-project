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
    def verify_location(self, location_reference: str, filename: str, line: int):
        response = self.dap_server.request_locations(location_reference)
        self.assertTrue(response["success"])
        self.assertTrue(response["body"]["source"]["path"].endswith(filename))
        self.assertEqual(response["body"]["line"], line)

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
        declaration_location_reference = locals["var1"].get(
            "declarationLocationReference"
        )
        self.assertIsNotNone(declaration_location_reference)
        self.verify_location(declaration_location_reference, "main.cpp", 11)
        value_location_reference = locals["var1"].get("valueLocationReference")
        self.assertIsNone(value_location_reference)

        # func_ptr has both a declaration and a valueLocation
        declaration_location_reference = locals["func_ptr"].get(
            "declarationLocationReference"
        )
        self.assertIsNotNone(declaration_location_reference)
        self.verify_location(declaration_location_reference, "main.cpp", 12)
        value_location_reference = locals["func_ptr"].get("valueLocationReference")
        self.assertIsNotNone(value_location_reference)
        self.verify_location(value_location_reference, "main.cpp", 3)

        # func_ref has both a declaration and a valueLocation
        declaration_location_reference = locals["func_ref"].get(
            "declarationLocationReference"
        )
        self.assertIsNotNone(declaration_location_reference)
        self.verify_location(declaration_location_reference, "main.cpp", 13)
        value_location_reference = locals["func_ref"].get("valueLocationReference")
        self.assertIsNotNone(value_location_reference)
        self.verify_location(value_location_reference, "main.cpp", 3)

        # member_ptr has both a declaration and a valueLocation
        declaration_location_reference = locals["member_ptr"].get(
            "declarationLocationReference"
        )
        self.assertIsNotNone(declaration_location_reference)
        self.verify_location(declaration_location_reference, "main.cpp", 14)
        value_location_reference = locals["member_ptr"].get("valueLocationReference")
        self.assertIsNotNone(value_location_reference)
        self.verify_location(value_location_reference, "main.cpp", 6)

        # virtual_member_ptr has a declarationLocation but no valueLocation
        declaration_location_reference = locals["virtual_member_ptr"].get(
            "declarationLocationReference"
        )
        self.assertIsNotNone(declaration_location_reference)
        self.verify_location(declaration_location_reference, "main.cpp", 15)
        value_location_reference = locals["virtual_member_ptr"].get(
            "valueLocationReference"
        )
        self.assertIsNone(value_location_reference)

        # `evaluate` responses for function pointers also have locations associated
        eval_res = self.dap_server.request_evaluate("greet")
        self.assertTrue(eval_res["success"])
        self.assertIn("valueLocationReference", eval_res["body"].keys())
