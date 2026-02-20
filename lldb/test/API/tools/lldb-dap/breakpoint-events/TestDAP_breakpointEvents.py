"""
Test lldb-dap setBreakpoints request
"""

from dap_server import Source
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import os


class TestDAP_breakpointEvents(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    def test_breakpoint_events(self):
        """
        This test follows the following steps.
        - Sets a breakpoint in a shared library from the preCommands.
        - Runs and stops at the entry point of a program.
        - Sets two new breakpoints, one in the main executable and one in the shared library
        - Both breakpoint is set but only the the shared library breakpoint is not verified yet.
        - We will then continue and expect to get a breakpoint event that
            informs us the breakpoint in the shared library has "changed"
            and the correct line number should be supplied.
        - We also verify the breakpoint set via the command interpreter should not
            be have breakpoint events sent back to VS Code as the UI isn't able to
            add new breakpoints to their UI.

        Code has been added that tags breakpoints set from VS Code
        DAP packets so we know the IDE knows about them. If VS Code is ever
        able to register breakpoints that aren't initially set in the GUI,
        then we will need to revise this.
        """
        main_source_path = self.getSourcePath("main.cpp")
        main_bp_line = line_number(main_source_path, "main breakpoint 1")
        main_source = Source.build(path=main_source_path)
        program = self.getBuildArtifact("a.out")

        # Set a breakpoint after creating the target by running a command line
        # command. It will eventually resolve and cause a breakpoint changed
        # event to be sent to lldb-dap. We want to make sure we don't send a
        # breakpoint any breakpoints that were set from the command line.
        # Breakpoints that are set via the VS code DAP packets will be
        # registered and marked with a special keyword to ensure we deliver
        # breakpoint events for these breakpoints but not for ones that are not
        # set via the command interpreter.

        shlib_env_key = self.platformContext.shlib_environment_var
        path_separator = self.platformContext.shlib_path_separator
        shlib_env_value = os.getenv(shlib_env_key)
        shlib_env_new_value = (
            self.getBuildDir()
            if shlib_env_value is None
            else (shlib_env_value + path_separator + self.getBuildDir())
        )

        # Set preCommand breakpoint
        func_unique_function_name = "unique_function_name"
        bp_command = f"breakpoint set --name {func_unique_function_name}"
        launch_seq = self.build_and_launch(
            program,
            preRunCommands=[bp_command],
            env={shlib_env_key: shlib_env_new_value},
        )
        self.dap_server.wait_for_event(["initialized"])
        dap_breakpoint_ids = []

        # We set the breakpoints after initialized event.
        # Set and verify new line breakpoint.
        response = self.dap_server.request_setBreakpoints(main_source, [main_bp_line])
        self.assertTrue(response["success"])
        breakpoints = response["body"]["breakpoints"]
        self.assertEqual(len(breakpoints), 1, "expects only one line breakpoint")
        main_breakpoint = breakpoints[0]
        main_bp_id = main_breakpoint["id"]
        dap_breakpoint_ids.append(main_bp_id)
        self.assertTrue(
            main_breakpoint["verified"], "expects main breakpoint to be verified"
        )

        # Set and verify new function breakpoint.
        func_foo = "foo"
        response = self.dap_server.request_setFunctionBreakpoints([func_foo])
        self.assertTrue(response["success"])
        breakpoints = response["body"]["breakpoints"]
        self.assertEqual(len(breakpoints), 1, "expects only one function breakpoint")
        func_foo_breakpoint = breakpoints[0]
        foo_bp_id = func_foo_breakpoint["id"]
        dap_breakpoint_ids.append(foo_bp_id)
        self.assertFalse(
            func_foo_breakpoint["verified"],
            "expects unique function breakpoint to not be verified",
        )

        self.dap_server.request_configurationDone()
        launch_response = self.dap_server.receive_response(launch_seq)
        self.assertIsNotNone(launch_response)
        self.assertTrue(launch_response["success"])

        # Wait for the next stop (breakpoint foo).
        self.verify_breakpoint_hit([foo_bp_id])
        unique_bp_id = 1

        # Check the breakpoints set in dap is verified.
        verified_breakpoint_ids = []
        events = self.dap_server.wait_for_breakpoint_events()
        for breakpoint_event in events:
            breakpoint_event_body = breakpoint_event["body"]
            if breakpoint_event_body["reason"] != "changed":
                continue
            breakpoint = breakpoint_event_body["breakpoint"]

            if "verified" in breakpoint_event_body:
                self.assertFalse(
                    breakpoint_event_body["verified"],
                    f"expects changed breakpoint to be verified. event: {breakpoint_event}",
                )
            id = breakpoint["id"]
            verified_breakpoint_ids.append(id)

        self.assertIn(main_bp_id, verified_breakpoint_ids)
        self.assertIn(foo_bp_id, verified_breakpoint_ids)
        self.assertNotIn(unique_bp_id, verified_breakpoint_ids)

        # Continue to the unique function breakpoint set from preRunCommands.
        unique_function_stop_event = self.continue_to_next_stop()[0]
        unique_body = unique_function_stop_event["body"]
        self.assertEqual(unique_body["reason"], "breakpoint")
        self.assertIn(unique_bp_id, unique_body["hitBreakpointIds"])

        # Clear line and function breakpoints and exit.
        self.dap_server.request_setFunctionBreakpoints([])
        self.dap_server.request_setBreakpoints(main_source, [])
        self.continue_to_exit()
