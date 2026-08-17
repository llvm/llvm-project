"""
Test lldb-dap dataBreakpointInfo and setDataBreakpoints requests
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import DataBreakpoint, LaunchArgs


@requireNotWasm("data breakpoints map to watchpoints")
class TestDAP_setDataBreakpoints(DAPTestCaseBase):
    ACCESS_TYPES = ["read", "write", "readWrite"]

    @skipIfWindows
    def test_duplicate_start_addresses(self):
        """Test setDataBreakpoints with multiple watchpoints starting at the same addresses."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = "main.cpp"
        first_loop_break_line = line_number(source, "// first loop breakpoint")
        with session.configure(LaunchArgs(program)) as ctx:
            session.resolve_source_breakpoints(source, [first_loop_break_line])
        stop_event = session.verify_stopped_on_breakpoint(after=ctx.process_event)

        # Verify write watchpoints on expressions `&x` and `arr+2`.
        top_frame_id = session.top_frame_from(stop_event).frame.id
        response_x = session.data_breakpoint_info("&x", 0, top_frame_id)
        response_arr_2 = session.data_breakpoint_info("arr+2", 0, top_frame_id)

        x_data_id = self.expect_not_none(response_x.body.dataId)
        arr_2_data_id = self.expect_not_none(response_arr_2.body.dataId)
        self.assertEqual(x_data_id.split("/")[1], "4")
        self.assertEqual(response_x.body.accessTypes, self.ACCESS_TYPES)
        self.assertEqual(arr_2_data_id.split("/")[1], "4")
        self.assertEqual(response_arr_2.body.accessTypes, self.ACCESS_TYPES)

        # The first breakpoint should be overwritten by the third breakpoint because
        # they share the same starting address. The debug adapter indicates this by
        # returning a breakpoint that is not verified for the first breakpoint.
        data_breakpoints = [
            DataBreakpoint(dataId=x_data_id, accessType="read"),
            DataBreakpoint(dataId=arr_2_data_id, accessType="write"),
            DataBreakpoint(dataId=x_data_id, accessType="write"),
        ]
        set_response = session.set_data_breakpoints(data_breakpoints)
        [bp_x_read, bp_arr_2, bp_x_write] = set_response.body.breakpoints
        self.assertFalse(bp_x_read.verified)
        self.assertTrue(bp_arr_2.verified)
        self.assertTrue(bp_x_write.verified)

        # Hit the write watchpoint on `x` at i == 1.
        stop_event = session.continue_to_breakpoint(self.expect_not_none(bp_x_write.id))
        top_frame = session.top_frame_from(stop_event)
        self.assertEqual(top_frame.locals["x"].value, "2")
        self.assertEqual(top_frame.locals["i"].value, "1")

        # Hit the write watchpoint on `arr[2]` at i == 2.
        stop_event = session.continue_to_breakpoint(self.expect_not_none(bp_arr_2.id))
        top_frame = session.top_frame_from(stop_event)
        self.assertEqual(top_frame.locals["arr"]["[2]"].value, "42")
        self.assertEqual(top_frame.locals["i"].value, "2")

        session.set_data_breakpoints([])
        session.continue_to_exit()

    @skipIfWindows
    def test_expression(self):
        """Tests setting data breakpoints on expression."""
        source = "main.cpp"
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        first_loop_break_line = line_number(source, "// first loop breakpoint")
        with session.configure(LaunchArgs(program)) as ctx:
            session.resolve_source_breakpoints(source, [first_loop_break_line])
        stop_event = session.verify_stopped_on_breakpoint(after=ctx.process_event)

        # Verify write watchpoints on expressions `&x` and `arr+2`.
        top_frame_id = session.top_frame_from(stop_event).frame.id
        response_x = session.data_breakpoint_info("&x", 0, top_frame_id)
        response_arr_2 = session.data_breakpoint_info("arr+2", 0, top_frame_id)

        x_data_id = self.expect_not_none(response_x.body.dataId)
        arr_2_data_id = self.expect_not_none(response_arr_2.body.dataId)
        self.assertEqual(x_data_id.split("/")[1], "4")
        self.assertEqual(response_x.body.accessTypes, self.ACCESS_TYPES)
        self.assertEqual(arr_2_data_id.split("/")[1], "4")
        self.assertEqual(response_arr_2.body.accessTypes, self.ACCESS_TYPES)

        data_breakpoints = [
            DataBreakpoint(dataId=x_data_id, accessType="write"),
            DataBreakpoint(dataId=arr_2_data_id, accessType="write"),
        ]
        set_response = session.set_data_breakpoints(data_breakpoints)
        [bp_x, bp_arr_2] = set_response.body.breakpoints
        self.assertTrue(bp_x.verified)
        self.assertTrue(bp_arr_2.verified)

        stop_event = session.continue_to_breakpoint(self.expect_not_none(bp_x.id))
        top_frame = session.top_frame_from(stop_event)
        self.assertEqual(top_frame.locals["x"].value, "2")
        self.assertEqual(top_frame.locals["i"].value, "1")

        stop_event = session.continue_to_breakpoint(self.expect_not_none(bp_arr_2.id))
        top_frame = session.top_frame_from(stop_event)
        self.assertEqual(top_frame.locals["arr"]["[2]"].value, "42")
        self.assertEqual(top_frame.locals["i"].value, "2")

        session.set_data_breakpoints([])
        session.continue_to_exit()

    @skipIfWindows
    def test_functionality(self):
        """Tests setting data breakpoints on variable."""
        source = "main.cpp"
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        first_loop_break_line = line_number(source, "// first loop breakpoint")
        with session.configure(LaunchArgs(program)) as ctx:
            session.resolve_source_breakpoints(source, [first_loop_break_line])
        stop_event = session.verify_stopped_on_breakpoint(after=ctx.process_event)

        top_frame_ctx = session.top_frame_from(stop_event)
        frame_id = top_frame_ctx.frame.id
        locals_ref = top_frame_ctx.locals.variablesReference

        # Verify write watchpoints on x and arr[2].
        response_x = session.data_breakpoint_info("x", locals_ref, frame_id)
        arr = top_frame_ctx.locals["arr"]
        arr_var_ref = self.expect_not_none(arr.variablesReference)
        response_arr_2 = session.data_breakpoint_info("[2]", arr_var_ref, frame_id)

        x_data_id = self.expect_not_none(response_x.body.dataId)
        arr_2_data_id = self.expect_not_none(response_arr_2.body.dataId)
        self.assertEqual(x_data_id.split("/")[1], "4")
        self.assertEqual(response_x.body.accessTypes, self.ACCESS_TYPES)
        self.assertEqual(arr_2_data_id.split("/")[1], "4")
        self.assertEqual(response_arr_2.body.accessTypes, self.ACCESS_TYPES)

        data_breakpoints = [
            DataBreakpoint(dataId=x_data_id, accessType="write"),
            DataBreakpoint(dataId=arr_2_data_id, accessType="write"),
        ]
        set_response = session.set_data_breakpoints(data_breakpoints)
        [bp_x, bp_arr_2] = set_response.body.breakpoints
        self.assertTrue(bp_x.verified)
        self.assertTrue(bp_arr_2.verified)

        stop_event = session.continue_to_breakpoint(self.expect_not_none(bp_x.id))
        top_frame = session.top_frame_from(stop_event)
        self.assertEqual(top_frame.locals["x"].value, "2")
        self.assertEqual(top_frame.locals["i"].value, "1")

        stop_event = session.continue_to_breakpoint(self.expect_not_none(bp_arr_2.id))
        top_frame = session.top_frame_from(stop_event)
        self.assertEqual(top_frame.locals["arr"]["[2]"].value, "42")
        self.assertEqual(top_frame.locals["i"].value, "2")

        session.set_data_breakpoints([])

        # Verify hit condition: skip past the second-loop breakpoint until `x`
        # has been written twice, then verify we stop with x == 3.
        second_loop_break_line = line_number(source, "// second loop breakpoint")
        breakpoint_ids = session.resolve_source_breakpoints(
            source, [second_loop_break_line]
        )
        session.continue_to_any_breakpoint(breakpoint_ids)
        set_response = session.set_data_breakpoints(
            [DataBreakpoint(dataId=x_data_id, accessType="write", hitCondition="2")]
        )
        [bp_hit] = set_response.body.breakpoints
        self.assertTrue(bp_hit.verified)
        stop_event = session.continue_to_breakpoint(self.expect_not_none(bp_hit.id))
        self.assertEqual(session.top_frame_from(stop_event).locals["x"].value, "3")

        # Test condition: only stop when the write makes x == 10.
        set_response = session.set_data_breakpoints(
            [DataBreakpoint(dataId=x_data_id, accessType="write", condition="x==10")]
        )
        [bp_cond] = set_response.body.breakpoints
        self.assertTrue(bp_cond.verified)
        stop_event = session.continue_to_breakpoint(self.expect_not_none(bp_cond.id))
        self.assertEqual(session.top_frame_from(stop_event).locals["x"].value, "10")

    @skipIfWindows
    def test_bytes(self):
        """Tests setting data breakpoints on memory range."""
        source = self.getSourcePath("main.cpp")
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        first_loop_break_line = line_number(source, "// first loop breakpoint")
        with session.configure(LaunchArgs(program)) as ctx:
            session.resolve_source_breakpoints(source, [first_loop_break_line])
        stop_event = session.verify_stopped_on_breakpoint(after=ctx.process_event)

        # Set write watchpoints on x and arr[2] using their memory references.
        top_frame = session.top_frame_from(stop_event)
        x_memory_reference = self.expect_not_none(top_frame.locals["x"].memoryReference)
        arr_2_mem_ref = self.expect_not_none(
            top_frame.locals["arr"]["[2]"].memoryReference
        )
        response_x = session.data_breakpoint_info_as_address(x_memory_reference, 4)
        response_arr_2 = session.data_breakpoint_info_as_address(arr_2_mem_ref, 4)

        x_data_id = self.expect_not_none(response_x.body.dataId)
        arr_2_data_id = self.expect_not_none(response_arr_2.body.dataId)
        self.assertEqual(x_data_id.split("/"), [x_memory_reference[2:], "4"])
        self.assertEqual(response_x.body.accessTypes, self.ACCESS_TYPES)
        self.assertEqual(arr_2_data_id.split("/"), [arr_2_mem_ref[2:], "4"])
        self.assertEqual(response_arr_2.body.accessTypes, self.ACCESS_TYPES)

        set_response = session.set_data_breakpoints(
            [
                DataBreakpoint(dataId=x_data_id, accessType="write"),
                DataBreakpoint(dataId=arr_2_data_id, accessType="write"),
            ]
        )
        [bp_x, bp_arr_2] = set_response.body.breakpoints
        self.assertTrue(bp_x.verified)
        self.assertTrue(bp_arr_2.verified)

        stop_event = session.continue_to_breakpoint(self.expect_not_none(bp_x.id))
        top_frame = session.top_frame_from(stop_event)
        self.assertEqual(top_frame.locals["x"].value, "2")
        self.assertEqual(top_frame.locals["i"].value, "1")

        stop_event = session.continue_to_breakpoint(self.expect_not_none(bp_arr_2.id))
        top_frame = session.top_frame_from(stop_event)
        self.assertEqual(top_frame.locals["arr"]["[2]"].value, "42")
        self.assertEqual(top_frame.locals["i"].value, "2")

        session.set_data_breakpoints([])
        session.continue_to_exit()
