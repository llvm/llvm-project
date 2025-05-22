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
        response_x = self.dap_server.request_dataBreakpointInfo("&x", 0)
        response_arr_2 = self.dap_server.request_dataBreakpointInfo("arr+2", 0)
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
        self.assertEqual(
            set_response["body"]["breakpoints"],
            [{"verified": False}, {"verified": True}, {"verified": True}],
        )

        self.continue_to_next_stop()
        x_val = self.dap_server.get_local_variable_value("x")
        i_val = self.dap_server.get_local_variable_value("i")
        self.assertEqual(x_val, "2")
        self.assertEqual(i_val, "1")

        self.continue_to_next_stop()
        arr_2 = self.dap_server.get_local_variable_child("arr", "[2]")
        i_val = self.dap_server.get_local_variable_value("i")
        self.assertEqual(arr_2["value"], "42")
        self.assertEqual(i_val, "2")

    @skipIfWindows
    def test_breakpoint_info_bytes(self):
        """Test supportBreakpointInfoBytes
        Set the watchpoint on `var` variable address + 6 characters.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        first_loop_break_line = line_number(source, "// first loop breakpoint")
        self.set_source_breakpoints(source, [first_loop_break_line])
        self.continue_to_next_stop()

        # get the address of `var` variable
        eval_response = self.dap_server.request_evaluate("&var", context="watch")
        self.assertTrue(eval_response["success"])
        var_address = eval_response["body"]["result"]

        var_byte_watch_size = 5
        bp_resp = self.dap_server.request_dataBreakpointInfo(
            var_address, asAddress=True, bytes_=var_byte_watch_size
        )
        resp_data_id = bp_resp["body"]["dataId"]
        self.assertTrue(
            bp_resp["success"], f"dataBreakpointInfo request failed: {bp_resp}"
        )
        self.assertEqual(resp_data_id.split("/")[1], str(var_byte_watch_size))

        data_breakpoints = [{"dataId": resp_data_id, "accessType": "write"}]
        self.dap_server.request_setDataBreakpoint(data_breakpoints)

        self.continue_to_breakpoint(breakpoint_id=1, is_watchpoint=True)
        eval_response = self.dap_server.request_evaluate("var", context="watch")
        self.assertTrue(eval_response["success"])
        var_value = eval_response["body"]["result"]
        self.assertEqual(var_value, '"HALLO"')

        # Remove the watchpoint because once it leaves this function scope, the address can be
        # be used by another variable or register.
        self.dap_server.request_setDataBreakpoint([])
        self.continue_to_exit()

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
        response_x = self.dap_server.request_dataBreakpointInfo("&x", 0)
        response_arr_2 = self.dap_server.request_dataBreakpointInfo("arr+2", 0)
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
        self.assertEqual(
            set_response["body"]["breakpoints"],
            [{"verified": True}, {"verified": True}],
        )

        self.continue_to_next_stop()
        x_val = self.dap_server.get_local_variable_value("x")
        i_val = self.dap_server.get_local_variable_value("i")
        self.assertEqual(x_val, "2")
        self.assertEqual(i_val, "1")

        self.continue_to_next_stop()
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
        # Test write watchpoints on x, arr[2]
        response_x = self.dap_server.request_dataBreakpointInfo("x", 1)
        arr = self.dap_server.get_local_variable("arr")
        response_arr_2 = self.dap_server.request_dataBreakpointInfo(
            "[2]", arr["variablesReference"]
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
        self.assertEqual(
            set_response["body"]["breakpoints"],
            [{"verified": True}, {"verified": True}],
        )

        self.continue_to_next_stop()
        x_val = self.dap_server.get_local_variable_value("x")
        i_val = self.dap_server.get_local_variable_value("i")
        self.assertEqual(x_val, "2")
        self.assertEqual(i_val, "1")

        self.continue_to_next_stop()
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
        self.assertEqual(set_response["body"]["breakpoints"], [{"verified": True}])
        self.continue_to_next_stop()
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
        self.assertEqual(set_response["body"]["breakpoints"], [{"verified": True}])
        self.continue_to_next_stop()
        x_val = self.dap_server.get_local_variable_value("x")
        self.assertEqual(x_val, "10")
