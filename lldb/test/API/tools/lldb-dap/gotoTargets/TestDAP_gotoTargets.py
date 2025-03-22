"""
Test lldb-dap gotoTarget request
"""

from typing import Dict, Any
from unittest import SkipTest

from lldbsuite.test.lldbtest import line_number
import lldbdap_testcase
import os


class TestDAP_gotoTargets(lldbdap_testcase.DAPTestCaseBase):
    def verify_variable(
        self, actual_dict: Dict[str, Any], expected_dict: Dict[str, Any]
    ):
        for key, value in expected_dict.items():
            actual_value = actual_dict[key]
            self.assertEqual(
                actual_value,
                value,
                f"values does not match for key: `{key}` expected_value: `{value}`, actual_value: `{actual_value}`",
            )

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
            response["success"], True, "request for gotoTargets should be successful"
        )
        target = response["body"]["targets"][0]
        self.assertGreaterEqual(
            target["id"], 0, "targetId should be greater than or equal to zero"
        )

        target_id = target["id"]
        thread_id = self.dap_server.get_thread_id()
        self.assertIsNotNone(thread_id, "threadId should not be none")

        response = self.dap_server.request_goto(thread_id, target_id)
        self.assertEqual(
            response["success"], True, "goto request with targetId should be successful"
        )

        stopped_events = self.dap_server.wait_for_stopped(timeout=0.200)
        is_goto = lambda event: event["body"]["reason"] == "goto"
        has_goto_event = any(map(is_goto, stopped_events))
        self.assertEqual(
            has_goto_event, True, "expected a stopped event with reason `goto`"
        )

        self.dap_server.request_next(thread_id)
        self.continue_to_next_stop()

        # Verify that `var_1=10` and `var_2=40`. This combination is only possible by
        # skipping execution of a line from the original program. Observing this combination
        # hence proves that our `goto` request actually skipped execution of the code line.
        var1_variable = self.dap_server.get_local_variable("var_1")
        var_1_expected = {
            "name": "var_1",
            "type": "int",
            "value": "10",
            "variablesReference": 0,
        }
        self.verify_variable(var1_variable, var_1_expected)

        var2_variable = self.dap_server.get_local_variable("var_2")
        var_2_expected = {
            "name": "var_2",
            "type": "int",
            "value": "40",
            "variablesReference": 0,
        }
        self.verify_variable(var2_variable, var_2_expected)

        self.continue_to_exit()

    def test_execute_again(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        source_file = "main.c"
        self.source_path = os.path.join(os.getcwd(), source_file)
        self.set_source_breakpoints(
            source_file, [line_number(source_file, "// breakpoint 2")]
        )
        self.continue_to_next_stop()

        end_var_3_value = self.dap_server.get_local_variable_value("var_3")
        self.assertEqual(end_var_3_value, "99")

        goto_line = line_number(source_file, "// goto 2")
        goto_column = 1
        response = self.dap_server.request_gotoTargets(
            source_file, self.source_path, goto_line, goto_column
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

        stopped_events = self.dap_server.wait_for_stopped(timeout=0.200)  # 200ms
        is_goto = lambda event: event["body"]["reason"] == "goto"
        has_goto_event = any(map(is_goto, stopped_events))
        self.assertEqual(has_goto_event, True, "expects stopped event with reason goto")

        self.dap_server.request_next(thread_id)
        self.continue_to_next_stop()

        goto_var_3_value = self.dap_server.get_local_variable_value("var_3")
        self.assertEqual(goto_var_3_value, "10")

        self.continue_to_next_stop()
        self.continue_to_exit()
