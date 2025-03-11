"""
Test lldb-dap gotoTarget request
"""

from lldbsuite.test.lldbtest import line_number
import lldbdap_testcase
import os


class TestDAP_gotoTarget(lldbdap_testcase.DAPTestCaseBase):
    def test_default(self):
        """
        Tests the jump to cursor of a simple program. No arguments,
        environment, or anything else is specified.
        This does not run any statement between the current breakpoint
        and the jump line location.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        source_file = "main.c"
        self.source_path = os.path.join(os.getcwd(), source_file)
        self.set_source_breakpoints(
            source_file, [line_number(source_file, "// breakpoint 1")]
        )
        self.continue_to_next_stop()

        first_var_1_object = self.dap_server.get_local_variable("var_1")
        self.assertEqual(first_var_1_object["value"], "10")

        goto_line = line_number(source_file, "// goto 1")
        goto_column = 1
        response = self.dap_server.request_gotoTargets(
            source_file, self.source_path, goto_line, goto_column
        )

        self.assertEqual(
            response["success"], True, "expects success when request for targets"
        )
        target = response["body"]["targets"][0]
        self.assertGreaterEqual(
            target["id"], 0, "targetId should be greater than or equal to zero"
        )

        target_id = target["id"]
        thread_id = self.dap_server.get_thread_id()
        self.assertIsNotNone(thread_id, "threadId should not be none")

        response = self.dap_server.request_goto(thread_id, target_id)

        self.assertEqual(response["success"], True, "expects success to go to targetId")

        self.dap_server.request_next(thread_id)
        self.continue_to_next_stop()

        local_variables = self.dap_server.get_local_variables()
        verify_variables = {
            "var_1": {
                "name": "var_1",
                "type": "int",
                "value": "10",
                "variablesReference": 0,
            },
            "var_2": {
                "name": "var_2",
                "type": "int",
                "value": "40",
                "variablesReference": 0,
            },
        }

        for variable in local_variables:
            name = variable["name"]
            verify_variable = verify_variables[name]

            for key, value in verify_variable.items():
                actual_value = variable[key]
                self.assertEqual(
                    actual_value,
                    value,
                    f"values does not match for key: `{key}` expected_value: `{value}`, actual_value: `{actual_value}`",
                )
        self.continue_to_exit()
