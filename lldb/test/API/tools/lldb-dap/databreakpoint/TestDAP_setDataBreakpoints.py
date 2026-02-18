"""
Test lldb-dap dataBreakpointInfo and setDataBreakpoints requests
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


class TestDAP_setDataBreakpoints(lldbdap_testcase.DAPTestCaseBase):
    def setUp(self):
        lldbdap_testcase.DAPTestCaseBase.setUp(self)
        self.accessTypes = ["read", "write", "readWrite"]

    @skipIfWindows
    def test_duplicate_start_addresses(self):
        """Test setDataBreakpoints with multiple watchpoints starting at the same addresses."""
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        first_loop_break_line = line_number(source, "// first loop breakpoint")
        self.set_source_breakpoints(source, [first_loop_break_line])
        self.continue_to_next_stop()
        self.dap_server.get_stackFrame()
        # Test setting write watchpoint using expressions: &x, arr+2
        response_x = self.dap_server.request_dataBreakpointInfo(0, "&x")
        response_arr_2 = self.dap_server.request_dataBreakpointInfo(0, "arr+2")
        # Test response from dataBreakpointInfo request.
        self.assertEqual(response_x["body"]["dataId"].split("/")[1], "4")
        self.assertEqual(response_x["body"]["accessTypes"], self.accessTypes)
        self.assertEqual(response_arr_2["body"]["dataId"].split("/")[1], "4")
        self.assertEqual(response_arr_2["body"]["accessTypes"], self.accessTypes)
        # The first one should be overwritten by the third one as they start at
        # the same address. This is indicated by returning {verified: False} for
        # the first one.
        dataBreakpoints = [
            {"dataId": response_x["body"]["dataId"], "accessType": "read"},
            {"dataId": response_arr_2["body"]["dataId"], "accessType": "write"},
            {"dataId": response_x["body"]["dataId"], "accessType": "write"},
        ]
        set_response = self.dap_server.request_setDataBreakpoint(dataBreakpoints)
        breakpoints = set_response["body"]["breakpoints"]
        self.assertEqual(len(breakpoints), 3)
        self.assertFalse(breakpoints[0]["verified"])
        self.assertTrue(breakpoints[1]["verified"])
        self.assertTrue(breakpoints[2]["verified"])

        self.dap_server.request_continue()
        self.verify_breakpoint_hit([breakpoints[2]["id"]])
        x_val = self.dap_server.get_local_variable_value("x")
        i_val = self.dap_server.get_local_variable_value("i")
        self.assertEqual(x_val, "2")
        self.assertEqual(i_val, "1")

        self.dap_server.request_continue()
        self.verify_breakpoint_hit([breakpoints[1]["id"]])
        arr_2 = self.dap_server.get_local_variable_child("arr", "[2]")
        i_val = self.dap_server.get_local_variable_value("i")
        self.assertEqual(arr_2["value"], "42")
        self.assertEqual(i_val, "2")

    @skipIfWindows
    def test_expression(self):
        """Tests setting data breakpoints on expression."""
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        first_loop_break_line = line_number(source, "// first loop breakpoint")
        self.set_source_breakpoints(source, [first_loop_break_line])
        self.continue_to_next_stop()
        self.dap_server.get_stackFrame()
        # Test setting write watchpoint using expressions: &x, arr+2
        response_x = self.dap_server.request_dataBreakpointInfo(0, "&x")
        response_arr_2 = self.dap_server.request_dataBreakpointInfo(0, "arr+2")
        # Test response from dataBreakpointInfo request.
        self.assertEqual(response_x["body"]["dataId"].split("/")[1], "4")
        self.assertEqual(response_x["body"]["accessTypes"], self.accessTypes)
        self.assertEqual(response_arr_2["body"]["dataId"].split("/")[1], "4")
        self.assertEqual(response_arr_2["body"]["accessTypes"], self.accessTypes)
        dataBreakpoints = [
            {"dataId": response_x["body"]["dataId"], "accessType": "write"},
            {"dataId": response_arr_2["body"]["dataId"], "accessType": "write"},
        ]
        set_response = self.dap_server.request_setDataBreakpoint(dataBreakpoints)
        breakpoints = set_response["body"]["breakpoints"]
        self.assertEqual(len(breakpoints), 2)
        self.assertTrue(breakpoints[0]["verified"])
        self.assertTrue(breakpoints[1]["verified"])

        self.dap_server.request_continue()
        self.verify_breakpoint_hit([breakpoints[0]["id"]])
        x_val = self.dap_server.get_local_variable_value("x")
        i_val = self.dap_server.get_local_variable_value("i")
        self.assertEqual(x_val, "2")
        self.assertEqual(i_val, "1")

        self.dap_server.request_continue()
        self.verify_breakpoint_hit([breakpoints[1]["id"]])
        arr_2 = self.dap_server.get_local_variable_child("arr", "[2]")
        i_val = self.dap_server.get_local_variable_value("i")
        self.assertEqual(arr_2["value"], "42")
        self.assertEqual(i_val, "2")

    @skipIfWindows
    def test_functionality(self):
        """Tests setting data breakpoints on variable."""
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        first_loop_break_line = line_number(source, "// first loop breakpoint")
        self.set_source_breakpoints(source, [first_loop_break_line])
        self.continue_to_next_stop()
        self.dap_server.get_local_variables()
        locals_ref = self.get_locals_scope_reference()
        self.assertIsNotNone(locals_ref, "Failed to get locals scope reference")
        # Test write watchpoints on x, arr[2]
        response_x = self.dap_server.request_dataBreakpointInfo(locals_ref, "x")
        arr = self.dap_server.get_local_variable("arr")
        response_arr_2 = self.dap_server.request_dataBreakpointInfo(
            arr["variablesReference"], "[2]"
        )

        # Test response from dataBreakpointInfo request.
        self.assertEqual(response_x["body"]["dataId"].split("/")[1], "4")
        self.assertEqual(response_x["body"]["accessTypes"], self.accessTypes)
        self.assertEqual(response_arr_2["body"]["dataId"].split("/")[1], "4")
        self.assertEqual(response_arr_2["body"]["accessTypes"], self.accessTypes)
        dataBreakpoints = [
            {"dataId": response_x["body"]["dataId"], "accessType": "write"},
            {"dataId": response_arr_2["body"]["dataId"], "accessType": "write"},
        ]
        set_response = self.dap_server.request_setDataBreakpoint(dataBreakpoints)
        breakpoints = set_response["body"]["breakpoints"]
        self.assertEqual(len(breakpoints), 2)
        self.assertTrue(breakpoints[0]["verified"])
        self.assertTrue(breakpoints[1]["verified"])

        self.dap_server.request_continue()
        self.verify_breakpoint_hit([breakpoints[0]["id"]])
        x_val = self.dap_server.get_local_variable_value("x")
        i_val = self.dap_server.get_local_variable_value("i")
        self.assertEqual(x_val, "2")
        self.assertEqual(i_val, "1")

        self.dap_server.request_continue()
        self.verify_breakpoint_hit([breakpoints[1]["id"]])
        arr_2 = self.dap_server.get_local_variable_child("arr", "[2]")
        i_val = self.dap_server.get_local_variable_value("i")
        self.assertEqual(arr_2["value"], "42")
        self.assertEqual(i_val, "2")
        self.dap_server.request_setDataBreakpoint([])

        # Test hit condition
        second_loop_break_line = line_number(source, "// second loop breakpoint")
        breakpoint_ids = self.set_source_breakpoints(source, [second_loop_break_line])
        self.continue_to_breakpoints(breakpoint_ids)
        dataBreakpoints = [
            {
                "dataId": response_x["body"]["dataId"],
                "accessType": "write",
                "hitCondition": "2",
            }
        ]
        set_response = self.dap_server.request_setDataBreakpoint(dataBreakpoints)
        breakpoints = set_response["body"]["breakpoints"]
        self.assertEqual(len(breakpoints), 1)
        self.assertTrue(breakpoints[0]["verified"])
        self.dap_server.request_continue()
        self.verify_breakpoint_hit([breakpoints[0]["id"]])
        x_val = self.dap_server.get_local_variable_value("x")
        self.assertEqual(x_val, "3")

        # Test condition
        dataBreakpoints = [
            {
                "dataId": response_x["body"]["dataId"],
                "accessType": "write",
                "condition": "x==10",
            }
        ]
        set_response = self.dap_server.request_setDataBreakpoint(dataBreakpoints)
        breakpoints = set_response["body"]["breakpoints"]
        self.assertEqual(len(breakpoints), 1)
        self.assertTrue(breakpoints[0]["verified"])
        self.dap_server.request_continue()
        self.verify_breakpoint_hit([breakpoints[0]["id"]])
        x_val = self.dap_server.get_local_variable_value("x")
        self.assertEqual(x_val, "10")

    @skipIfWindows
    def test_bytes(self):
        """Tests setting data breakpoints on memory range."""
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        first_loop_break_line = line_number(source, "// first loop breakpoint")
        self.set_source_breakpoints(source, [first_loop_break_line])
        self.continue_to_next_stop()
        # Test write watchpoints on x, arr[2]
        x = self.dap_server.get_local_variable("x")
        response_x = self.dap_server.request_dataBreakpointInfo(
            0, x["memoryReference"], 4
        )
        arr_2 = self.dap_server.get_local_variable_child("arr", "[2]")
        response_arr_2 = self.dap_server.request_dataBreakpointInfo(
            0, arr_2["memoryReference"], 4
        )

        # Test response from dataBreakpointInfo request.
        self.assertEqual(
            response_x["body"]["dataId"].split("/"), [x["memoryReference"][2:], "4"]
        )
        self.assertEqual(response_x["body"]["accessTypes"], self.accessTypes)
        self.assertEqual(
            response_arr_2["body"]["dataId"].split("/"),
            [arr_2["memoryReference"][2:], "4"],
        )
        self.assertEqual(response_arr_2["body"]["accessTypes"], self.accessTypes)
        dataBreakpoints = [
            {"dataId": response_x["body"]["dataId"], "accessType": "write"},
            {"dataId": response_arr_2["body"]["dataId"], "accessType": "write"},
        ]
        set_response = self.dap_server.request_setDataBreakpoint(dataBreakpoints)
        breakpoints = set_response["body"]["breakpoints"]
        self.assertEqual(len(breakpoints), 2)
        self.assertTrue(breakpoints[0]["verified"])
        self.assertTrue(breakpoints[1]["verified"])

        self.dap_server.request_continue()
        self.verify_breakpoint_hit([breakpoints[0]["id"]])
        x_val = self.dap_server.get_local_variable_value("x")
        i_val = self.dap_server.get_local_variable_value("i")
        self.assertEqual(x_val, "2")
        self.assertEqual(i_val, "1")

        self.dap_server.request_continue()
        self.verify_breakpoint_hit([breakpoints[1]["id"]])
        arr_2 = self.dap_server.get_local_variable_child("arr", "[2]")
        i_val = self.dap_server.get_local_variable_value("i")
        self.assertEqual(arr_2["value"], "42")
        self.assertEqual(i_val, "2")
        self.dap_server.request_setDataBreakpoint([])
