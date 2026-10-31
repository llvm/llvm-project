"""
Test lldb-dap runInTerminal reverse request
"""

import sys
from typing import cast

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase, DAPTestSession
from lldbsuite.test.tools.lldb_dap.types import *


@skipIfBuildType(["debug"])
@skipIfWasm  # runInTerminal has the client run the program, and a Wasm module is not executable
class TestDAP_runInTerminal(DAPTestCaseBase):
    @skipIfAsan
    def test_launch_with_args_and_env(self):
        """
        Tests the "runInTerminal" reverse request. It makes sure that the IDE can
        launch the inferior with the correct environment variables and arguments.
        """
        program = self.getBuildArtifact("a.out")
        source = "main.c"
        session = self.build_and_create_session()

        launch_args = LaunchArgs(
            program,
            console=Console.INTEGRATED_TERMINAL,
            args=["no_cmd"],
            env=["FOO=bar"],
        )
        with session.configure(launch_args) as ctx:
            breakpoint_line = line_number(source, "// breakpoint")
            [bp_id] = session.resolve_source_breakpoints(source, [breakpoint_line])

        request_args = self.run_in_terminal_args(session)
        self.assertIn(self.lldb_dap_exe, request_args.args)
        self.assertIn(program, request_args.args)
        self.assertIn("no_cmd", request_args.args)
        request_env = self.expect_not_none(request_args.env)
        self.assertIn("FOO", request_env)
        if sys.platform == "win32":
            self.assertIn("_NO_DEBUG_HEAP", request_env)

        stop = session.verify_stopped_on_breakpoint(bp_id, after=ctx.process_event)
        frame = session.top_frame_from(stop)

        # Verify process stopped in the main function.
        self.assertEqual(frame.locals["counter"].value_as_int, 1)

        # We verify we were able to set the launch arguments.
        self.assertEqual(frame.locals["argc"].value_as_int, 2)
        self.assertIn("no_cmd", frame.evaluate("argv[1]").result)

        # We verify we were able to set the environment.
        self.assertIn("bar", frame.evaluate("foo").result)

        if sys.platform == "win32":
            self.assertIn('"1"', frame.evaluate("nodebugheap").result)

        session.continue_to_exit()

    @skipIfAsan
    def test_launch_with_object_env(self):
        """
        Tests the "runInTerminal" reverse request. It makes sure that the IDE can
        launch the inferior with the correct environment variables using an object.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        process_event = session.launch(
            LaunchArgs(program, console=Console.INTEGRATED_TERMINAL, env={"FOO": "BAR"})
        )

        request_env = self.expect_not_none(self.run_in_terminal_args(session).env)
        self.assertIn("FOO", request_env)
        self.assertEqual("BAR", request_env["FOO"])

        session.verify_process_exited(after=process_event)

    def test_launch_invalid_program_fails(self):
        program = self.getBuildArtifact("INVALIDPROGRAM")
        session = self.create_session()
        session.initialize_sequence(session.initialize_args)
        launch_handle = session.send_request(
            LaunchArgs(
                program,
                console=Console.INTEGRATED_TERMINAL,
                args=["echo", "foobar"],
                env=["FOO=bar"],
            )
        )
        session.verify_configuration_done(expected_success=False)

        response = launch_handle.error()
        response_body = self.expect_not_none(response.body)
        error_msg = self.expect_not_none(response_body.error)
        self.assertIn(f"'{program}' does not exist", error_msg.format)

        # Verify we never received an initialized event.
        with self.assertRaises(TimeoutError):
            session.wait_for_earliest_event(InitializedEvent, timeout=0)

    def test_client_missing_run_in_terminal_feature(self):
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        session.update_initialize_args(supportsRunInTerminalRequest=False)

        launch_handle = session.initialize_and_launch(
            LaunchArgs(program, console=Console.INTEGRATED_TERMINAL)
        )
        session.verify_configuration_done(expected_success=False)

        err_response = launch_handle.error()
        response_body = self.expect_not_none(err_response.body)
        error_msg = self.expect_not_none(response_body.error)
        self.assertIn("Client does not support RunInTerminal.", error_msg.format)

    def run_in_terminal_args(self, session: DAPTestSession) -> RunInTerminalArgs:
        request = session.last_reverse_request()
        self.assertIsInstance(request, RunInTerminalRequest)
        return cast(RunInTerminalRequest, request).arguments
