"""
Test lldb-dap setFunctionBreakpoints request
"""

import os

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import (
    DAPTestGetTargetBreakpointsArgs,
    FunctionBreakpoint,
    LaunchArgs,
    ModuleById,
    ModuleByName,
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
    def test_modules(self):
        """Tests that a function breakpoint is identified by its function name
        and its set of modules (in any order), and that the modules, identified
        by name or by id, restrict the lookup."""
        session = self.build_and_create_session()
        # The binary is built from a copy of main.cpp.
        main_path = os.path.realpath(self.getBuildArtifact("main-copy.cpp"))
        with session.configure(LaunchArgs(self.getBuildArtifact("a.out"))) as ctx:
            bp_ids = session.resolve_source_breakpoints(
                main_path, [line_number("main.cpp", "// break loop")]
            )
        session.verify_stopped_on_breakpoint(bp_ids, after=ctx.process_event)

        # `twelve` is defined in a.out and `foo` in libother, which is loaded
        # by now.
        modules = session.get_modules()
        a_out = modules["a.out"]
        other = next(m for name, m in modules.items() if name.startswith("libother"))

        a_out_name = ModuleByName(a_out.name)
        other_name = ModuleByName(other.name)
        a_out_id = ModuleById(a_out.id)
        other_id = ModuleById(other.id)

        def set_breakpoints(*breakpoints):
            response = session.set_function_breakpoints(list(breakpoints))
            self.assertEqual(len(response.body.breakpoints), len(breakpoints))
            return response.body.breakpoints

        def target_breakpoint_ids():
            # Ignore the source breakpoint that stopped the process.
            response = session.send_request(DAPTestGetTargetBreakpointsArgs()).result()
            return [bp.id for bp in response.body.breakpoints if bp.id not in bp_ids]

        # The order of the modules and duplicates don't matter.
        [bp1, bp2, bp3] = set_breakpoints(
            FunctionBreakpoint(name="twelve", modules=[a_out_name, other_name]),
            FunctionBreakpoint(
                name="twelve", modules=[other_name, a_out_name, a_out_name]
            ),
            FunctionBreakpoint(name="twelve", modules=[a_out_name]),
        )
        self.assertTrue(bp1.verified)
        self.assertEqual(bp1.id, bp2.id)
        # A different set of modules is a different breakpoint.
        self.assertNotEqual(bp1.id, bp3.id)
        self.assertTrue(bp3.verified)
        self.assertCountEqual(target_breakpoint_ids(), [bp1.id, bp3.id])

        # Dropping a breakpoint only removes that one.
        [bp] = set_breakpoints(
            FunctionBreakpoint(name="twelve", modules=[other_name, a_out_name])
        )
        self.assertEqual(bp.id, bp1.id)
        self.assertEqual(target_breakpoint_ids(), [bp1.id])

        # No (non-empty) module is the same as no restriction.
        [bp1, bp2, bp3, bp4] = set_breakpoints(
            FunctionBreakpoint(name="foo"),
            FunctionBreakpoint(name="foo", modules=[]),
            FunctionBreakpoint(name="foo", modules=[ModuleByName("")]),
            FunctionBreakpoint(name="foo", modules=[ModuleById("")]),
        )
        self.assertTrue(bp1.verified)
        self.assertEqual({bp1.id, bp2.id, bp3.id, bp4.id}, {bp1.id})
        self.assertEqual(target_breakpoint_ids(), [bp1.id])

        # The module names restrict the lookup.
        [in_a_out, in_other, nowhere] = set_breakpoints(
            FunctionBreakpoint(name="foo", modules=[a_out_name]),
            FunctionBreakpoint(name="foo", modules=[other_name]),
            FunctionBreakpoint(name="foo", modules=[ModuleByName("does-not-exist")]),
        )
        self.assertFalse(in_a_out.verified)
        self.assertTrue(in_other.verified)
        self.assertFalse(nowhere.verified)
        self.assertEqual(len({in_a_out.id, in_other.id, nowhere.id}), 3)

        # The module ids restrict the lookup, and they are case insensitive.
        [in_a_out, in_other, in_other_lower, in_both, in_both_reversed] = (
            set_breakpoints(
                FunctionBreakpoint(name="foo", modules=[a_out_id]),
                FunctionBreakpoint(name="foo", modules=[other_id]),
                FunctionBreakpoint(
                    name="foo", modules=[ModuleById(other.id.lower())]
                ),
                FunctionBreakpoint(name="foo", modules=[a_out_id, other_id]),
                FunctionBreakpoint(name="foo", modules=[other_id, a_out_id]),
            )
        )
        self.assertFalse(in_a_out.verified)
        self.assertTrue(in_other.verified)
        self.assertEqual(in_other.id, in_other_lower.id)
        self.assertTrue(in_both.verified)
        self.assertEqual(in_both.id, in_both_reversed.id)
        self.assertEqual(len({in_a_out.id, in_other.id, in_both.id}), 3)

        # An id that doesn't match a module must not leave the breakpoint
        # unrestricted.
        unknown_id = "00000000-0000-0000-0000-000000000001"
        [unknown, invalid] = set_breakpoints(
            FunctionBreakpoint(name="twelve", modules=[ModuleById(unknown_id)]),
            FunctionBreakpoint(name="twelve", modules=[ModuleById("not-a-uuid")]),
        )
        for bp in (unknown, invalid):
            self.assertFalse(bp.verified)
            self.assertIn("module id not found", bp.message)
        # No breakpoint was created in the target.
        self.assertEqual(target_breakpoint_ids(), [])

        # Names and ids can be mixed: the lookup is restricted to all of them.
        # An id that isn't found doesn't make the others fail.
        [mixed, mixed_reversed, mixed_unknown, same_modules] = set_breakpoints(
            FunctionBreakpoint(name="foo", modules=[a_out_id, other_name]),
            FunctionBreakpoint(name="foo", modules=[other_name, a_out_id]),
            FunctionBreakpoint(
                name="foo", modules=[ModuleById(unknown_id), other_name]
            ),
            FunctionBreakpoint(name="foo", modules=[a_out_name, other_name]),
        )
        self.assertTrue(mixed.verified)
        self.assertEqual(mixed.id, mixed_reversed.id)
        self.assertTrue(mixed_unknown.verified)
        self.assertTrue(same_modules.verified)
        # The identity depends on how the modules are identified.
        self.assertEqual(
            len({mixed.id, mixed_unknown.id, same_modules.id}),
            3,
        )
        self.assertCountEqual(
            target_breakpoint_ids(), [mixed.id, mixed_unknown.id, same_modules.id]
        )

        # Mixed with `twelve`, which is only defined in a.out: it is found
        # through the id.
        [twelve] = set_breakpoints(
            FunctionBreakpoint(name="twelve", modules=[a_out_id, other_name])
        )
        self.assertTrue(twelve.verified)

        # Remove all the breakpoints so the process runs to the end.
        session.set_function_breakpoints([])
        session.set_source_breakpoints(main_path, [])
        session.continue_to_exit()

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
