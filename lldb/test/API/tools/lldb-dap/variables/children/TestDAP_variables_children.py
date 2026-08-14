from lldbsuite.test.decorators import expectedFailureAll
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap.dap_types import LaunchArgs
from lldbsuite.test.tools.lldb_dap.lldb_dap_testcase import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.session_helpers import ExpectVar


class TestDAP_variables_children(DAPTestCaseBase):
    def test_get_num_children(self):
        """Test that GetNumChildren is not called for formatters not producing indexed children."""
        session = self.build_and_create_session()
        program = self.getBuildArtifact("a.out")

        launch_args = LaunchArgs(
            program,
            preRunCommands=[
                f"command script import '{self.getSourcePath('formatter.py')}'"
            ],
        )
        with session.configure(launch_args) as ctx:
            source = self.getSourcePath("main.cpp")
            [bp_id] = session.resolve_source_breakpoints(
                source, [line_number(source, "// break here")]
            )

        stopped_event = session.verify_stopped_on_breakpoint(
            bp_id, after=ctx.process_event
        )
        thread = session.thread_context_from(stopped_event)
        local_vars = thread.top_frame().locals.variables()
        indexed_var = next(x for x in local_vars if x.name == "indexed")
        not_indexed_var = next(x for x in local_vars if x.name == "not_indexed")

        self.assertIsNotNone(indexed_var.indexedVariables)
        self.assertEqual(indexed_var.indexedVariables, 1)
        self.assertIsNone(not_indexed_var.indexedVariables)

        resp_body = session.evaluate(
            "`script formatter.num_children_calls", context="repl"
        )
        self.assertIn("['Indexed']", resp_body.result)

    @expectedFailureAll(archs=["arm$", "arm64", "aarch64"])
    def test_return_variable_with_children(self):
        """
        Test the stepping out of a function with return value show the children correctly
        """
        session = self.build_and_create_session()
        program = self.getBuildArtifact("a.out")

        with session.configure(LaunchArgs(program)) as ctx:
            function_name = "test_return_variable_with_children"
            [func_bp_id] = session.resolve_function_breakpoints([function_name])

        stopped_event = session.verify_stopped_on_breakpoint(
            func_bp_id, after=ctx.process_event
        )

        thread_ctx = session.thread_context_from(stopped_event)
        thread_ctx.step_out()

        local_variables = thread_ctx.top_frame().locals.variables()
        self.assertIsNot(len(local_variables), 0)
        return_variable = local_variables[0].variable
        self.assertEqual(return_variable.name, "(Return Value)")

        result_var_ref = return_variable.variablesReference
        self.assertIsNot(result_var_ref, None, "There is no result value")

        result_children = session.get_variables(result_var_ref)
        verify_children = {"buffer": '"hello world!"', "x": "10", "y": "20"}
        for child in result_children:
            verify_value = verify_children.get(child.name)
            self.assertNotEqual(verify_value, None)
            self.assertEqual(
                child.value, verify_value, "Expected child value does not match"
            )
