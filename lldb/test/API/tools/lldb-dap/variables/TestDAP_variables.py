"""
Test lldb-dap variables request
"""

import os
from typing import List, Optional

from lldbsuite.test import lldbplatformutil
from lldbsuite.test.decorators import (
    no_debug_info_test,
    skipIfAsan,
    skipIfWindows,
    skipUnlessDarwin,
)
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap.types import (
    EvaluateContext,
    LaunchArgs,
    VariablesArgs,
)
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase, ExpectEval, ExpectVar


def make_expected_buffer(start_idx, count, offset=0):
    return {
        f"[{i}]": ExpectVar(type="int", value=str(i + offset))
        for i in range(start_idx, start_idx + count)
    }


class TestDAP_variables(DAPTestCaseBase):
    SHARED_BUILD_TESTCASE = False

    def darwin_dwarf_missing_obj(self, initCommands: Optional[List[str]]):
        self.build(debug_info="dwarf")
        program = self.getBuildArtifact("a.out")
        main_obj = self.getBuildArtifact("main.o")
        self.assertTrue(os.path.exists(main_obj))

        # Delete the main.o file that contains the debug info so we force an
        # error when we run to main and try to get variables.
        os.unlink(main_obj)
        self.assertTrue(os.path.exists(program), "executable must exist")

        session = self.create_session()
        with session.configure(
            LaunchArgs(program=program, initCommands=initCommands)
        ) as ctx:
            breakpoint_ids = session.resolve_function_breakpoints(["main"])
            self.assertEqual(len(breakpoint_ids), 1, "expect one breakpoint")

        stop_event = session.verify_stopped_on_breakpoint(
            breakpoint_ids, after=ctx.process_event
        )
        thread_id = self.expect_not_none(stop_event.body.threadId)
        frame = session.thread_context_from(thread_id).top_frame()

        var_args = VariablesArgs(variablesReference=frame.locals.variablesReference)
        error_response = session.send_request(var_args).error()
        error_body = self.expect_not_none(error_response.body)
        error_message = self.expect_not_none(error_body.error)
        self.assertEqual(
            f'debug map object file "{main_obj}" containing debug info does not exist, debug info will not be loaded',
            error_message.format,
        )
        self.assertTrue(error_message.showUser)

    def do_test_scopes_variables_setVariable_evaluate(
        self, enableAutoVariableSummaries: bool
    ):
        """Tests the "scopes", "variables", "setVariable", and "evaluate" packets."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = "main.cpp"
        breakpoint1_line = line_number(source, "// breakpoint 1")
        breakpoint2_line = line_number(source, "// breakpoint 2")
        breakpoint3_line = line_number(source, "// breakpoint 3")

        launch_args = LaunchArgs(
            program=program, enableAutoVariableSummaries=enableAutoVariableSummaries
        )
        with session.configure(launch_args) as ctx:
            breakpoint_ids = session.resolve_source_breakpoints(
                source, [breakpoint1_line, breakpoint2_line, breakpoint3_line]
            )

        bp1, bp2, bp3 = breakpoint_ids
        stop_event = session.verify_stopped_on_breakpoint(bp1, after=ctx.process_event)
        thread_id = self.expect_not_none(stop_event.body.threadId)
        frame = session.top_frame_from(thread_id)
        local_vars = session.get_variables(frame.locals.variablesReference)
        global_vars = session.get_variables(frame.globals.variablesReference)

        buffer_children = make_expected_buffer(0, 16)
        expect_locals = {
            "argc": ExpectVar(type="int", value="1"),
            "argv": ExpectVar(type="const char **", startswith="0x", has_var_ref=True),
            "pt": ExpectVar(
                type="PointType",
                has_var_ref=True,
                read_only=True,
                children={
                    "x": ExpectVar(type="int", value="11"),
                    "y": ExpectVar(type="int", value="22"),
                    "buffer": ExpectVar(read_only=True, children=buffer_children),
                },
            ),
            "valid_str": ExpectVar(),
            "malformed_str": ExpectVar(),
            "x": ExpectVar(type="int"),
        }

        s_global = ExpectVar(type="int", value="234")
        g_global = ExpectVar(type="int", value="123")
        expect_globals = {
            "s_local": ExpectVar(type="float", value="2.25"),
        }
        if lldbplatformutil.getHostPlatform() == "windows":
            expect_globals["::s_global"] = s_global
            expect_globals["g_global"] = g_global
        else:
            expect_globals["s_global"] = s_global
            expect_globals["::g_global"] = g_global

        session.verify_variables(local_vars, expect_locals)
        session.verify_variables(global_vars, expect_globals)

        pt_var = frame.locals["pt"]
        pt_buffer = pt_var["buffer"]

        # We need to test the functionality of the "variables" request as it
        # has optional parameters like "start" and "count" to limit the number
        # of variables that are fetched.
        var_ref = pt_buffer.variablesReference
        children = session.get_variables(var_ref)
        session.verify_variables(children, buffer_children)
        # start=0 still gets all children.
        children = session.get_variables(var_ref, start=0)
        session.verify_variables(children, buffer_children)
        # count=0 gets all children.
        children = session.get_variables(var_ref, count=0)
        session.verify_variables(children, buffer_children)
        # An oversized count gets all children, no more.
        children = session.get_variables(var_ref, count=1000)
        session.verify_variables(children, buffer_children)
        # start and count gets only the children we want.
        children = session.get_variables(var_ref, start=5, count=5)
        session.verify_variables(children, make_expected_buffer(5, 5))
        # An out-of-range start gets an empty list.
        children = session.get_variables(var_ref, start=32, count=1)
        self.assertEqual(
            len(children), 0, "verify we get no variables back for an invalid start"
        )

        # Test evaluate.
        if enableAutoVariableSummaries:
            pt_summary = "{x:11, y:22, buffer:{...}}"
            buf_summary = "{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...}"
        else:
            pt_summary = "PointType"
            buf_summary = "int[16]"

        expressions = {
            "pt.x": ExpectEval(type="int", result="11", has_var_ref=False),
            "pt.buffer[2]": ExpectEval(type="int", result="2", has_var_ref=False),
            "pt": ExpectEval(type="PointType", startswith=pt_summary, has_var_ref=True),
            "pt.buffer": ExpectEval(
                type="int[16]",
                startswith=buf_summary,
                has_var_ref=True,
            ),
            "argv": ExpectEval(type="const char **", startswith="0x", has_var_ref=True),
            "argv[0]": ExpectEval(
                type="const char *", startswith="0x", has_var_ref=True
            ),
            "2+3": ExpectEval(type="int", result="5", has_var_ref=False),
        }
        for expression, expected in expressions.items():
            expr_result = frame.evaluate(expression)
            session.verify_evaluate(expr_result, expected)

        # Test setting variables.
        self.expect_success(frame.locals.set("argc", 123))
        argc = frame.locals["argc"].value_as_int
        self.assertEqual(argc, 123, f"verify argc was set to 123 (123 != {argc})")

        self.expect_success(frame.locals.set("argv", 0x1234))
        argv = frame.locals["argv"].value_as_int
        self.assertEqual(
            argv, 0x1234, f"verify argv was set to 0x1234 (0x1234 != {argv:#x})"
        )

        # Test hexadecimal format.
        hex_set = self.expect_success(frame.locals.set("argc", 42, is_hex=True))
        self.assertEqual(hex_set.body.type, "int")
        self.assertEqual(hex_set.body.value, "0x0000002a")
        self.expect_success(frame.locals.set("argc", 123))

        # Set a variable whose name is synthetic (an array index) and verify
        # the value by reading it back.
        variable_value = 100
        set_result = self.expect_success(pt_buffer.set("[0]", variable_value))
        self.assertEqual(set_result.body.type, "int")
        self.assertEqual(set_result.body.value, str(variable_value))

        pt_buffer_ref = pt_buffer.variablesReference
        children = session.get_variables(pt_buffer_ref, start=0, count=1)
        session.verify_variables(children, make_expected_buffer(0, 1, variable_value))
        # Update the new value in the buffer expectations so subsequent
        # verifications stay consistent.
        buffer_children["[0]"].value = str(variable_value)

        # Set a variable whose name is a real child value (e.g. "pt.x") and
        # verify the value by reading it back.
        pt_var_ref = pt_var.variablesReference
        self.expect_success(pt_var.set("x", "g_global - 12"))
        children = session.get_variables(pt_var_ref, start=0, count=1)
        value = children[0].value
        self.assertEqual(value, "111", f"verify pt.x got set to 111 (111 != {value})")

        # Continue to the second breakpoint to check shadowed variables and the new
        # local variables is update to the new values.
        session.continue_to_breakpoint(bp2)
        frame = session.top_frame_from(thread_id)

        expect_locals["argc"].value = "123"
        # Build a child dict with `pt.x` updated to its new value.
        pt_children = self.expect_not_none(expect_locals["pt"].children)
        pt_children["x"].value = "111"
        expect_locals["x @ main.cpp:27"] = ExpectVar(type="int", value="89")
        expect_locals["x @ main.cpp:29"] = ExpectVar(type="int", value="42")
        expect_locals["x @ main.cpp:31"] = ExpectVar(type="int", value="72")

        local_variables = session.get_variables(frame.locals.variablesReference)
        session.verify_variables(local_variables, expect_locals)

        # Renaming a variable with and without the differentiator suffix.
        self.expect_error(frame.locals.set("x2", 9))
        self.expect_error(frame.locals.set("x @ main.cpp:0", 9))

        self.expect_success(frame.locals.set("x @ main.cpp:27", 19))
        self.expect_success(frame.locals.set("x @ main.cpp:29", 21))
        self.expect_success(frame.locals.set("x @ main.cpp:31", 23))

        # An invalid value should have no effect.
        self.expect_error(frame.locals.set("x @ main.cpp:31", "invalid"))

        expect_locals["x @ main.cpp:27"].value = "19"
        expect_locals["x @ main.cpp:29"].value = "21"
        expect_locals["x @ main.cpp:31"].value = "23"

        local_vars = session.get_variables(frame.locals.variablesReference)
        session.verify_variables(local_vars, expect_locals)

        # The plain `x` variable should refer to the innermost x.
        self.expect_success(frame.locals.set("x", 22))
        expect_locals["x @ main.cpp:31"].value = "22"

        local_vars = session.get_variables(frame.locals.variablesReference)
        session.verify_variables(local_vars, expect_locals)

        # At breakpoint 3 there should be no shadowed variables.
        session.continue_to_breakpoint(bp3)
        frame = session.top_frame_from(thread_id)

        local_vars = session.get_variables(frame.locals.variablesReference)
        names = [var.name for var in local_vars]
        # The first shadowed `x` shouldn't have a suffix anymore.
        expect_locals["x"] = ExpectVar(type="int", value="19")
        self.assertNotIn("x @ main.cpp:27", names)
        self.assertNotIn("x @ main.cpp:29", names)
        self.assertNotIn("x @ main.cpp:31", names)

        session.verify_variables(local_vars, expect_locals)
        session.continue_to_exit()

    @skipIfWindows
    def test_scopes_variables_setVariable_evaluate(self):
        self.do_test_scopes_variables_setVariable_evaluate(
            enableAutoVariableSummaries=False
        )

    @skipIfWindows
    def test_scopes_variables_setVariable_evaluate_with_descriptive_summaries(
        self,
    ):
        self.do_test_scopes_variables_setVariable_evaluate(
            enableAutoVariableSummaries=True
        )

    @skipIfWindows
    def do_test_scopes_and_evaluate_expansion(self, enableAutoVariableSummaries: bool):
        """Test that an evaluated expression expands successfully after "scopes"
        packets, and that permanent expressions persist across resumes."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = "main.cpp"

        launch_args = LaunchArgs(
            program=program, enableAutoVariableSummaries=enableAutoVariableSummaries
        )
        with session.configure(launch_args) as ctx:
            lines = [
                line_number(source, "// breakpoint 1"),
                line_number(source, "// breakpoint 3"),
                line_number(source, "// breakpoint 6"),
                line_number(source, "// breakpoint 7"),
                line_number(source, "// breakpoint 8"),
            ]
            breakpoint_ids = session.resolve_source_breakpoints(source, lines)

        bp1, bp3, bp6, bp7, bp8 = breakpoint_ids
        stop_event = session.verify_stopped_on_breakpoint(bp1, after=ctx.process_event)
        thread_id = self.expect_not_none(stop_event.body.threadId)
        frame = session.thread_context_from(thread_id).top_frame()

        # Verify locals.
        local_vars = session.get_variables(frame.locals.variablesReference)
        buffer_children = make_expected_buffer(0, 32)
        expect_locals: dict[str, ExpectVar] = {
            "argc": ExpectVar(type="int", value="1", has_indexed_variables=False),
            "argv": ExpectVar(
                type="const char **",
                startswith="0x",
                has_var_ref=True,
                has_indexed_variables=False,
            ),
            "pt": ExpectVar(
                type="PointType",
                has_var_ref=True,
                read_only=True,
                has_indexed_variables=False,
                children={
                    "x": ExpectVar(type="int", value="11", has_indexed_variables=False),
                    "y": ExpectVar(type="int", value="22", has_indexed_variables=False),
                    "buffer": ExpectVar(
                        indexed_variables=16,
                        read_only=True,
                        children=buffer_children,
                    ),
                },
            ),
            "valid_str": ExpectVar(
                type="const char *",
                matches=r'0x\w+ "𐌶𐌰L𐌾𐍈 C𐍈𐌼𐌴𐍃"',
            ),
            "malformed_str": ExpectVar(
                type="const char *",
                matches=r'0x\w+ "lone trailing \\x81\\x82 bytes"',
            ),
            "x": ExpectVar(type="int", has_indexed_variables=False),
        }
        session.verify_variables(local_vars, expect_locals)

        expandable_name = "pt"
        if enableAutoVariableSummaries:
            pt_summary = "{x:11, y:22, buffer:{...}}"
        else:
            pt_summary = "PointType"
        repl_result = """(PointType) $0 = {
  x = 11
  y = 22
  buffer = {
    [0] = 0
    [1] = 1
    [2] = 2
    [3] = 3
    [4] = 4
    [5] = 5
    [6] = 6
    [7] = 7
    [8] = 8
    [9] = 9
    [10] = 10
    [11] = 11
    [12] = 12
    [13] = 13
    [14] = 14
    [15] = 15
  }
}"""

        expandable_children: dict[str, ExpectVar] = {
            "x": ExpectVar(type="int", value="11"),
            "y": ExpectVar(type="int", value="22"),
            "buffer": ExpectVar(read_only=True, children=buffer_children),
        }
        expected_contexts_evals: dict[EvaluateContext, ExpectEval] = {
            "repl": ExpectEval(
                type="PointType",
                result=repl_result,
                has_var_ref=True,
                has_indexed_variables=False,
                children=expandable_children,
            ),
            "hover": ExpectEval(
                type="PointType",
                startswith=pt_summary,
                has_var_ref=True,
                has_indexed_variables=False,
                children=expandable_children,
            ),
            "watch": ExpectEval(
                type="PointType",
                startswith=pt_summary,
                has_var_ref=True,
                has_indexed_variables=False,
                children=expandable_children,
            ),
            "variables": ExpectEval(
                type="PointType",
                startswith=pt_summary,
                has_var_ref=True,
                has_indexed_variables=False,
                children=expandable_children,
            ),
        }

        # Evaluate from each known context.
        permanent_expandable_ref: Optional[int] = None
        temporary_expandable_ref: Optional[int] = None
        for context, expected_eval in expected_contexts_evals.items():
            result = session.evaluate(
                expandable_name, frameId=frame.id, context=context
            )

            if context == "repl":  # Save the variablesReference.
                permanent_expandable_ref = result.variablesReference
            else:
                temporary_expandable_ref = result.variablesReference

            session.verify_evaluate(result, expected_eval)

        # Evaluate locals again.
        local_vars = session.get_variables(frame.locals.variablesReference)
        session.verify_variables(local_vars, expect_locals)

        # The previously evaluated expression should still expand.
        var_ref = self.expect_not_none(temporary_expandable_ref)
        session.verify_variables(session.get_variables(var_ref), expandable_children)

        session.continue_to_breakpoint(bp6)
        frame = session.top_frame_from(thread_id)
        if enableAutoVariableSummaries:
            my_var_value = '{name:"hello world!", x:42, y:7}'
        else:
            my_var_value = "(unnamed struct)"
        session.verify_variable(
            frame.locals["my_var"].variable,
            ExpectVar(
                type="(unnamed struct)",
                value=my_var_value,
                evaluate_name="my_var",
                read_only=True,
            ),
        )

        session.continue_to_breakpoint(bp7)
        frame = session.top_frame_from(thread_id)

        home_var = frame.locals["home"]
        session.verify_variable(
            home_var.variable,
            ExpectVar(
                type="MySock",
                value="MySock",
                evaluate_name="home",
                read_only=True,
            ),
        )

        anon_var = home_var["(anonymous)"]
        if enableAutoVariableSummaries:
            anon_value_re = r"{ipv4:.*, ipv6:.*}"
        else:
            anon_value_re = r"MySock::\(anonymous union\)"
        session.verify_variable(
            anon_var.variable,
            ExpectVar(
                type="MySock::(anonymous union)",
                matches=anon_value_re,
                read_only=True,
                has_evaluate_name=False,
            ),
        )
        inner_union_vars = session.get_variables(anon_var.variablesReference)
        session.verify_variables(
            inner_union_vars,
            {
                "ipv4": ExpectVar(
                    type="unsigned char[4]",
                    evaluate_name="home.ipv4",
                    read_only=True,
                ),
                "ipv6": ExpectVar(
                    type="unsigned char[6]",
                    evaluate_name="home.ipv6",
                    read_only=True,
                ),
            },
        )

        session.continue_to_breakpoint(bp8)
        frame = session.top_frame_from(thread_id)

        if enableAutoVariableSummaries:
            e_value = "{lo:10, hi:11}"
        else:
            e_value = "example"
        e_var = frame.locals["e"]
        session.verify_variable(
            e_var.variable,
            ExpectVar(
                type="example",
                value=e_value,
                evaluate_name="e",
                read_only=True,
            ),
        )
        inner_bitfields_struct = session.get_variables(e_var.variablesReference)
        session.verify_variables(
            inner_bitfields_struct,
            {
                "lo": ExpectVar(
                    type="unsigned int",
                    value="10",
                    evaluate_name="e.lo",
                    variables_reference=0,
                ),
                "(anonymous)": ExpectVar(type="int", value="0", variables_reference=0),
                "hi": ExpectVar(
                    type="unsigned int",
                    value="11",
                    evaluate_name="e.hi",
                    variables_reference=0,
                ),
            },
        )

        # Continue to breakpoint 3. The permanent variable should still exist after the resume.
        session.continue_to_breakpoint(bp3)
        frame = session.top_frame_from(thread_id)

        permanent_ref = self.expect_not_none(permanent_expandable_ref)
        permanent_ref_children = session.get_variables(permanent_ref)
        session.verify_variables(permanent_ref_children, expandable_children)

        # The frame's scopes should carry corresponding presentation hints.
        self.assertEqual(frame.globals.scope.name, "Globals")
        self.assertEqual(frame.locals.scope.name, "Locals")
        self.assertEqual(frame.locals.scope.presentationHint, "locals")
        self.assertEqual(frame.registers.scope.name, "Registers")
        self.assertEqual(frame.registers.scope.presentationHint, "registers")

        # An invalid variablesReference should produce a clear error.
        for wrong_var_ref in (-6000, -1, 4000):
            var_args = VariablesArgs(variablesReference=wrong_var_ref)
            response = session.send_request(var_args).error()

            error_body = self.expect_not_none(response.body)
            error_msg = self.expect_not_none(error_body.error)
            self.assertTrue(
                error_msg.format.startswith("invalid variablesReference"),
                f"seen error message: {error_msg.format}",
            )

    def test_scopes_and_evaluate_expansion(self):
        self.do_test_scopes_and_evaluate_expansion(enableAutoVariableSummaries=False)

    def test_scopes_and_evaluate_expansion_with_descriptive_summaries(self):
        self.do_test_scopes_and_evaluate_expansion(enableAutoVariableSummaries=True)

    def do_test_indexedVariables(self, enableSyntheticChildDebugging: bool):
        """Test that arrays and `lldb.SBValue` objects with synthetic child
        providers expose `indexedVariables`. This lets the IDE avoid fetching
        too many children at once."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = "main.cpp"
        breakpoint_line = line_number(source, "// breakpoint 4")

        launch_args = LaunchArgs(
            program=program,
            enableSyntheticChildDebugging=enableSyntheticChildDebugging,
        )
        with session.configure(launch_args) as ctx:
            breakpoint_ids = session.resolve_source_breakpoints(
                source, [breakpoint_line]
            )

        stop_event = session.verify_stopped_on_breakpoint(
            breakpoint_ids, after=ctx.process_event
        )
        thread_id = self.expect_not_none(stop_event.body.threadId)
        frame = session.thread_context_from(thread_id).top_frame()

        # Verify locals. Vector variables can have one extra entry from the
        # fake "[raw]" child.
        local_vars = session.get_variables(frame.locals.variablesReference)
        raw_child_count = 1 if enableSyntheticChildDebugging else 0
        expect_locals: dict[str, ExpectVar] = {
            "small_array": ExpectVar(indexed_variables=5, read_only=True),
            "large_array": ExpectVar(indexed_variables=200, read_only=True),
            "small_vector": ExpectVar(
                indexed_variables=5 + raw_child_count, read_only=True
            ),
            "large_vector": ExpectVar(
                indexed_variables=200 + raw_child_count, read_only=True
            ),
            "pt": ExpectVar(read_only=True, has_indexed_variables=False),
        }
        session.verify_variables(local_vars, expect_locals)

        # Verify we produce a "[raw]" fake child carrying the real SBValue
        # for the synthetic type.
        expect_children: dict[str, ExpectVar] = {
            f"[{i}]": ExpectVar(type="int", value="0") for i in range(5)
        }
        if enableSyntheticChildDebugging:
            expect_children["[raw]"] = ExpectVar(
                type="std::vector<int>", value="size=5", read_only=True
            )

        small_vector = frame.locals["small_vector"]
        children = session.get_variables(small_vector.variablesReference)
        session.verify_variables(children, expect_children)

        if enableSyntheticChildDebugging:
            raw_child = next(c for c in children if c.name == "[raw]")
            self.assertEqual(
                raw_child.evaluateName,
                "small_vector",
                "'evaluateName' for '[raw]' field should be the original variable name.",
            )
            raw_children = session.get_variables(raw_child.variablesReference)
            self.assertGreater(
                len(raw_children),
                0,
                "Expected std::vector to contain a raw underlying value with internal properties.",
            )

    @skipIfWindows
    def test_return_variables(self):
        """Stepping out of a function with a return value should expose the
        returned value as a local."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()

        function_name = "test_return_variable"
        with session.configure(LaunchArgs(program=program)) as ctx:
            [bp_id] = session.resolve_function_breakpoints([function_name])

        stop_event = session.verify_stopped_on_breakpoint(
            bp_id, after=ctx.process_event
        )
        thread_ctx = session.thread_context_from(stop_event)
        thread_ctx.step_out()

        frame = thread_ctx.top_frame()
        local_vars = session.get_variables(frame.locals.variablesReference)

        return_name = "(Return Value)"
        expect_locals: dict[str, ExpectVar] = {
            return_name: ExpectVar(type="int", value="300", read_only=True),
            "argc": ExpectVar(),
            "argv": ExpectVar(),
            "pt": ExpectVar(read_only=True),
            "valid_str": ExpectVar(),
            "malformed_str": ExpectVar(),
            "x": ExpectVar(),
            "return_result": ExpectVar(type="int"),
        }
        session.verify_variables(local_vars, expect_locals)

        self.expect_error(frame.locals.set("(Return Value)", 20))

    @skipIfWindows
    def test_indexedVariables(self):
        self.do_test_indexedVariables(enableSyntheticChildDebugging=False)

    @skipIfWindows
    def test_indexedVariables_with_raw_child_for_synthetics(self):
        self.do_test_indexedVariables(enableSyntheticChildDebugging=True)

    @skipIfWindows
    @skipIfAsan  # FIXME this fails with a non-asan issue on green dragon.
    def test_registers(self):
        """Test that registers whose byte size is the size of a pointer on
        the current system get formatted as `lldb::eFormatAddressInfo`. The
        formatted value should be a pointer followed by a description of the
        address. To test this we look for the PC value in the general purpose
        registers, and since we will be stopped in main.cpp, verify that its
        value starts with a pointer and is followed by a description that
        contains main.cpp."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        with session.configure(LaunchArgs(program)) as ctx:
            source = "main.cpp"
            breakpoint1_line = line_number(source, "// breakpoint 1")
            breakpoint_ids = session.resolve_source_breakpoints(
                source, [breakpoint1_line]
            )
        stop_event = session.verify_stopped_on_breakpoint(
            breakpoint_ids, after=ctx.process_event
        )

        pc_name: Optional[str] = None
        arch = self.getArchitecture()
        if arch in ("x86", "x86_64"):
            pc_name = "rip"
        elif arch.startswith("arm"):
            pc_name = "pc"

        if pc_name is None:
            self.skipTest(f"unknown program counter name for architecture: {arch}")

        # Verify locals.
        top_frame = session.top_frame_from(stop_event)
        gpr_reg_set = top_frame.registers["General Purpose Registers"]
        pc_reg = gpr_reg_set[pc_name].variable

        self.assertTrue(pc_reg.value.startswith("0x"))
        self.assertIn("a.out`main + ", pc_reg.value)
        self.assertIn("at main.cpp:", pc_reg.value)

    @no_debug_info_test
    @skipUnlessDarwin
    def test_darwin_dwarf_missing_obj(self):
        """
        Test that if we build a binary with DWARF in .o files and we remove
        the .o file for main.cpp, that we get a variable named "<error>"
        whose value matches the appropriate error. Errors when getting
        variables are returned in the LLDB API when the user should be
        notified of issues that can easily be solved by rebuilding or
        changing compiler options and are designed to give better feedback
        to the user.
        """
        self.darwin_dwarf_missing_obj(None)

    @no_debug_info_test
    @skipUnlessDarwin
    def test_darwin_dwarf_missing_obj_with_symbol_ondemand_enabled(self):
        """
        Test that if we build a binary with DWARF in .o files and we remove
        the .o file for main.cpp, that we get a variable named "<error>"
        whose value matches the appropriate error. Test with symbol.load-on-demand enabled.
        """
        initCommands = ["settings set symbols.load-on-demand true"]
        self.darwin_dwarf_missing_obj(initCommands)

    @no_debug_info_test
    @skipIfWindows
    def test_value_format(self):
        """Test that toggling a variable's value format between decimal and
        hexadecimal works."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        with session.configure(LaunchArgs(program)) as ctx:
            source = "main.cpp"
            breakpoint1_line = line_number(source, "// breakpoint 1")
            breakpoint_ids = session.resolve_source_breakpoints(
                source, [breakpoint1_line]
            )
        stop_event = session.verify_stopped_on_breakpoint(
            breakpoint_ids, after=ctx.process_event
        )

        thread_ctx = session.thread_context_from(stop_event)
        top_frame = thread_ctx.top_frame()
        # Verify locals with decimal value format.
        var_pt = top_frame.locals.with_format(is_hex=False)["pt"]
        self.assertEqual(var_pt["x"].value, "11")
        self.assertEqual(var_pt["y"].value, "22")

        # Verify locals with hex value format.
        var_pt = top_frame.locals.with_format(is_hex=True)["pt"]
        self.assertEqual(var_pt["x"].value, "0x0000000b")
        self.assertEqual(var_pt["y"].value, "0x00000016")

        # Toggle back and verify decimal value format again.
        var_pt = top_frame.locals.with_format(is_hex=False)["pt"]
        self.assertEqual(var_pt["x"].value, "11")
        self.assertEqual(var_pt["y"].value, "22")

    @skipIfWindows
    def test_variable_id_uniqueness_simple(self):
        """
        Simple regression test for variable ID uniqueness across frames.
        Ensures variable IDs are not reused between different scopes/frames.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        with session.configure(LaunchArgs(program)) as ctx:
            source = "main.cpp"
            bp_line = line_number(source, "// breakpoint 3")
            [bp_id] = session.resolve_source_breakpoints(source, [bp_line])

        stop_event = session.verify_stopped_on_breakpoint(
            bp_id, after=ctx.process_event
        )
        frames = session.thread_context_from(stop_event).frames()
        self.assertGreaterEqual(len(frames), 2, "need at least 2 frames")

        # Collect scope reference that has children.
        scope_refs = [
            scope.variablesReference
            for frame in frames[:3]
            for scope in frame.scopes()
            if scope.variablesReference != 0
        ]
        self.assertGreater(len(scope_refs), 0, "should have found variable references")

        seen_refs: set[int] = set()
        # Verify scope references are unique.
        for scope_ref in scope_refs:
            self.assertNotIn(scope_ref, seen_refs, f"{scope_ref=} was reused!")
            seen_refs.add(scope_ref)

        # Verify variable references are unique.
        for scope_ref in scope_refs:
            for var in session.get_variables(scope_ref):
                var_ref = var.variablesReference
                if var_ref != 0:
                    self.assertNotIn(
                        var_ref,
                        seen_refs,
                        f"variable {var.name} {var_ref=} was reused!",
                    )
                    seen_refs.add(var_ref)
