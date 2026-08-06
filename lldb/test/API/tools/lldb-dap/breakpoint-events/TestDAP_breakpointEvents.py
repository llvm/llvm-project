"""
Test lldb-dap setBreakpoints request
"""

import os
from typing import List

from lldbsuite.test.decorators import (
    skipIfTargetDoesNotSupportSharedLibraries,
    skipIfWindows,
)
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import (
    Breakpoint,
    BreakpointEvent,
    BreakpointReason,
    Event,
    LaunchArgs,
)


@skipIfTargetDoesNotSupportSharedLibraries()
class TestDAP_breakpointEvents(DAPTestCaseBase):
    def collect_breakpoint_changed_events(self, session, *, after, until: Event):
        """Collect every changed BreakpointEvent recorded in (after ..= until) inclusive."""
        self.assertLess(after.seq, until.seq)
        events: List[Breakpoint] = []

        def visit(evt: Event) -> bool:
            if isinstance(evt, BreakpointEvent):
                if evt.body.reason == BreakpointReason.CHANGED:
                    events.append(evt.body.breakpoint)
            return evt.seq >= until.seq

        session.wait_for_any_event(
            (BreakpointEvent, type(until)), after=after, until=visit
        )
        return events

    @skipIfWindows
    def test_breakpoint_events(self):
        """
        This test follows the following steps.
        - Sets a breakpoint in a shared library using the preRunCommands.
        - Sets two new breakpoints, a line breakpoint in the main executable and a function
          breakpoint on `foo` (defined in the shared library).
        - The main breakpoint is not verified but the foo breakpoint is
            unverified initially (the shared library isn't loaded yet).
        - After the shared library loads, both DAP breakpoints emit
            `breakpoint` events with reason=`changed`. The command-line
            breakpoint set via preRunCommands must NOT emit any event,
            because the IDE didn't ask for it.

        Code has been added that tags breakpoints set from VS Code
        DAP packets so we know the IDE knows about them. If VS Code is ever
        able to register breakpoints that aren't initially set in the GUI,
        then we will need to revise this.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        main_source_path = self.getSourcePath("main.cpp")
        main_bp_line = line_number(main_source_path, "main breakpoint 1")

        shlib_env_key = self.platformContext.shlib_environment_var
        path_separator = self.platformContext.shlib_path_separator
        shlib_env_value = os.getenv(shlib_env_key)
        shlib_env_new_value = (
            self.getBuildDir()
            if shlib_env_value is None
            else (shlib_env_value + path_separator + self.getBuildDir())
        )

        # Set a breakpoint via the command interpreter. lldb-dap tags DAP-set
        # breakpoints with a marker. events from command-line breakpoints (like
        # this one) must not be sent back to the client.
        unique_function = "unique_function_name"
        bp_command = f"breakpoint set --name {unique_function}"
        unique_bp_id = 1

        launch_args = LaunchArgs(
            program=program,
            preRunCommands=[bp_command],
            env={shlib_env_key: shlib_env_new_value},
        )
        with session.configure(launch_args) as ctx:
            [main_bp_id] = session.resolve_source_breakpoints(
                main_source_path, [main_bp_line]
            )

            # Set a function breakpoint on foo (in the shared library).
            # It must arrive unverified because the shlib isn't loaded yet.
            foo_resp = session.set_function_breakpoints(["foo"])
            self.assertEqual(
                len(foo_resp.body.breakpoints),
                1,
                "expects only one function breakpoint",
            )
            foo_bp = foo_resp.body.breakpoints[0]
            foo_bp_id = self.expect_not_none(foo_bp.id)
            self.assertFalse(
                foo_bp.verified,
                "expects unique function breakpoint to not be verified",
            )

        # First stop: foo breakpoint hit (after the shared library loaded).
        foo_stop = session.verify_stopped_on_breakpoint(
            foo_bp_id, after=ctx.process_event
        )

        # Collect every 'BreakpointEvent' with reason 'changed' between 'InitalizeResponse
        # and and the 'foo breakpoint stop'.
        # The IDE-tracked breakpoints (main + foo) should emit breakpoint `changed` events.
        evt_breakpoints = self.collect_breakpoint_changed_events(
            session, after=ctx.init_response, until=foo_stop
        )

        # A breakpoint may emit several `changed` events as it progresses
        # (e.g. unverified -> verified once the shlib loads). Keep the last
        # observed state per breakpoint id. The final state for DAP-tracked breakpoints
        # must be verified, and the command-line breakpoint must not appear.
        last_state = {bp.id: bp for bp in evt_breakpoints}

        for bp_id, name in [(main_bp_id, "main"), (foo_bp_id, "foo")]:
            final_bp = self.expect_not_none(
                last_state.get(bp_id),
                f"expected a changed event for {name} breakpoint {bp_id}",
            )
            self.assertTrue(
                final_bp.verified, f"{name} breakpoint must finish verified: {final_bp}"
            )

        self.assertNotIn(
            unique_bp_id,
            last_state,
            "command-line breakpoint must not emit any event",
        )

        # Continue to the unique_function breakpoint (set via preRunCommands).
        unique_func_stop = session.continue_to_next_stop()
        self.assertEqual(unique_func_stop.body.reason, "breakpoint")
        self.assertIn(unique_bp_id, unique_func_stop.body.hitBreakpointIds or [])

        # Clear line and function breakpoints and exit.
        session.set_function_breakpoints([])
        session.set_source_breakpoints(main_source_path, [])
        session.continue_to_exit()
