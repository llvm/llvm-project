import os

import dap_server
import lldbdap_testcase
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestDAP_variables_children(lldbdap_testcase.DAPTestCaseBase):
    def test_get_num_children(self):
        """Test that GetNumChildren is not called for formatters not producing indexed children."""
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(
            program,
            preRunCommands=[
                "command script import '%s'" % self.getSourcePath("formatter.py")
            ],
        )
        source = "main.cpp"
        breakpoint1_line = line_number(source, "// break here")
        lines = [breakpoint1_line]

        breakpoint_ids = self.set_source_breakpoints(
            source, [line_number(source, "// break here")]
        )
        self.continue_to_breakpoints(breakpoint_ids)

        local_vars = self.dap_server.get_local_variables()
        print(local_vars)
        indexed = next(filter(lambda x: x["name"] == "indexed", local_vars))
        not_indexed = next(filter(lambda x: x["name"] == "not_indexed", local_vars))
        self.assertIn("indexedVariables", indexed)
        self.assertEqual(indexed["indexedVariables"], 1)
        self.assertNotIn("indexedVariables", not_indexed)

        self.assertIn(
            "['Indexed']",
            self.dap_server.request_evaluate(
                "`script formatter.num_children_calls", context="repl"
            )["body"]["result"],
        )

    @skipIf(archs=["arm", "arm64", "aarch64"])
    def test_return_variable_with_children(self):
        """
        Test the stepping out of a function with return value show the children correctly
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        function_name = "test_return_variable_with_children"
        breakpoint_ids = self.set_function_breakpoints([function_name])

        self.assertEqual(len(breakpoint_ids), 1)
        self.continue_to_breakpoints(breakpoint_ids)

        threads = self.dap_server.get_threads()
        for thread in threads:
            if thread.get("reason") == "breakpoint":
                thread_id = thread.get("id")
                self.assertIsNot(thread_id, None)

                self.stepOut(threadId=thread_id)

                local_variables = self.dap_server.get_local_variables()

                # verify has return variable as local
                result_variable = list(
                    filter(
                        lambda val: val.get("name") == "(Return Value)", local_variables
                    )
                )
                self.assertEqual(len(result_variable), 1)
                result_variable = result_variable[0]

                result_var_ref = result_variable.get("variablesReference")
                self.assertIsNot(result_var_ref, None, "There is no result value")

                result_value = self.dap_server.request_variables(result_var_ref)
                result_children = result_value["body"]["variables"]
                self.assertNotEqual(
                    result_children, None, "The result does not have children"
                )

                verify_children = {"buffer": '"hello world!"', "x": "10", "y": "20"}
                for child in result_children:
                    actual_name = child["name"]
                    actual_value = child["value"]
                    verify_value = verify_children.get(actual_name)
                    self.assertNotEqual(verify_value, None)
                    self.assertEqual(
                        actual_value,
                        verify_value,
                        "Expected child value does not match",
                    )

                break
