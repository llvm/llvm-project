"""
Test lldb-dap setFunctionBreakpoints request
"""

from lldbsuite.test.decorators import (
    skipIfTargetDoesNotSupportSharedLibraries,
    skipIfWindows,
)
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import (
    DAPTestGetTargetBreakpointsArgs,
    FunctionBreakpoint,
    LaunchArgs,
)


@skipIfTargetDoesNotSupportSharedLibraries()
class TestDAP_setFunctionBreakpoints(DAPTestCaseBase):
    @skipIfWindows
    def test_set_and_clear(self):
        """Tests setting and clearing function breakpoints.
        This packet is a bit tricky on the debug adapter side since there
        is no "clearFunction Breakpoints" packet. Function breakpoints
        are set by sending a "setFunctionBreakpoints" packet with zero or
        more function names. If function breakpoints have been set before,
        any existing breakpoints must remain set, and any new breakpoints
        must be created, and any breakpoints that were in previous requests
        and are not in the current request must be removed. This function
        tests this setting and clearing and makes sure things happen
        correctly. It doesn't test hitting breakpoints and the functionality
        of each breakpoint, like 'conditions' and 'hitCondition' settings.
        """
        # Visual Studio Code Debug Adapters have no way to specify the file
        # without launching or attaching to a process, so we must start a
        # process in order to be able to set breakpoints.
        session = self.build_and_create_session()
        with session.configure(LaunchArgs(self.getBuildArtifact("a.out"))):
            functions = ["twelve"]
            # Set a function breakpoint at 'twelve'.
            response = session.set_function_breakpoints(functions)
            breakpoints = response.body.breakpoints
            self.assertEqual(
                len(breakpoints),
                len(functions),
                f"expect {len(functions)} source breakpoints",
            )
            bp_id_12 = self.expect_not_none(breakpoints[0].id)
            self.assertTrue(breakpoints[0].verified, "expect breakpoint verified")

            # Add an extra name and make sure we have two breakpoints after this.
            functions.append("thirteen")
            response = session.set_function_breakpoints(functions)
            breakpoints = response.body.breakpoints
            self.assertEqual(
                len(breakpoints),
                len(functions),
                f"expect {len(functions)} source breakpoints",
            )
            for bp in breakpoints:
                self.assertTrue(bp.verified, "expect breakpoint verified")

            # There is no breakpoint delete packet, clients just send another
            # setFunctionBreakpoints packet with the different function names.
            functions.remove("thirteen")
            response = session.set_function_breakpoints(functions)
            breakpoints = response.body.breakpoints
            self.assertEqual(
                len(breakpoints),
                len(functions),
                f"expect {len(functions)} source breakpoints",
            )
            for bp in breakpoints:
                self.assertEqual(
                    bp.id, bp_id_12, 'verify "twelve" breakpoint ID is same'
                )
                self.assertTrue(bp.verified, "expect breakpoint still verified")

            # Now get the full list of breakpoints set in the target and verify
            # we have only 1 breakpoints set. The response above could have told
            # us about 1 breakpoints, but we want to make sure we don't have the
            # second one still set in the target.
            response = session.send_request(DAPTestGetTargetBreakpointsArgs()).result()
            breakpoints = response.body.breakpoints
            self.assertEqual(
                len(breakpoints),
                len(functions),
                f"expect {len(functions)} source breakpoints",
            )
            for bp in breakpoints:
                self.assertEqual(
                    bp.id, bp_id_12, 'verify "twelve" breakpoint ID is same'
                )
                self.assertTrue(bp.verified, "expect breakpoint still verified")

            # Now clear all breakpoints for the source file by passing down an
            # empty lines array.
            functions = []
            response = session.set_function_breakpoints(functions)
            breakpoints = response.body.breakpoints
            self.assertEqual(
                len(breakpoints),
                len(functions),
                f"expect {len(functions)} source breakpoints",
            )

            # Verify with the target that all breakpoints have been cleared.
            response = session.send_request(DAPTestGetTargetBreakpointsArgs()).result()
            breakpoints = response.body.breakpoints
            self.assertEqual(
                len(breakpoints),
                len(functions),
                f"expect {len(functions)} source breakpoints",
            )

    @skipIfWindows
    def test_functionality(self):
        """Tests hitting breakpoints and the functionality of a single
        breakpoint, like 'conditions' and 'hitCondition' settings."""
        session = self.build_and_create_session()
        program = self.getBuildArtifact("a.out")

        # Set a breakpoint on "twelve" with no condition and no hitCondition.
        with session.configure(LaunchArgs(program)) as ctx:
            [bp_id] = session.resolve_function_breakpoints(["twelve"])

        # Verify we hit the breakpoint we just set.
        stop_event = session.verify_stopped_on_breakpoint(
            bp_id, after=ctx.process_event
        )

        # Make sure i is zero at first breakpoint.
        thread_ctx = session.thread_context_from(stop_event)
        i = thread_ctx.top_frame().locals["i"].value_as_int
        self.assertEqual(i, 0, "i != 0 after hitting breakpoint")

        # Update the condition on our breakpoint.
        func_bp = FunctionBreakpoint(name="twelve", condition="i==4")
        [condition_bp_id] = session.resolve_function_breakpoints([func_bp])
        self.assertEqual(
            bp_id,
            condition_bp_id,
            "existing breakpoint should have its condition updated",
        )

        session.continue_to_breakpoint(bp_id)
        i = thread_ctx.top_frame().locals["i"].value_as_int
        self.assertEqual(i, 4, "i != 4 showing conditional works")

        func_bp = FunctionBreakpoint(name="twelve", hitCondition="2")
        [hit_condition_bp_id] = session.resolve_function_breakpoints([func_bp])
        self.assertEqual(
            bp_id,
            hit_condition_bp_id,
            "existing breakpoint should have its condition updated",
        )

        # Continue with a hitCondition of 2 and expect it to skip 1 value.
        session.continue_to_breakpoint(bp_id)
        i = thread_ctx.top_frame().locals["i"].value_as_int
        self.assertEqual(i, 6, "i != 6 showing hitCondition works")

        # Continue after hitting our hitCondition and make sure it only goes
        # up by 1.
        session.continue_to_breakpoint(bp_id)
        i = thread_ctx.top_frame().locals["i"].value_as_int
        self.assertEqual(i, 7, "i != 7 showing post hitCondition hits every time")
