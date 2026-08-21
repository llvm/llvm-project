"""
Test lldb-dap setBreakpoints request
"""

import os
import shutil
from typing import Dict

from lldbsuite.test.decorators import (
    skipIfTargetDoesNotSupportSharedLibraries,
    skipIfWasm,
    skipIfWindows,
)
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import (
    DAPTestGetTargetBreakpointsArgs,
    LaunchArgs,
    SetBreakpointsArgs,
    Source,
    SourceBreakpoint,
)


@skipIfWasm  # inferior built without exception support
class TestDAP_setBreakpoints(DAPTestCaseBase):
    SHARED_BUILD_TESTCASE = False

    def setUp(self):
        DAPTestCaseBase.setUp(self)

        self.main_basename = "main-copy.cpp"
        self.main_path = os.path.realpath(self.getBuildArtifact(self.main_basename))

    @skipIfTargetDoesNotSupportSharedLibraries()
    @skipIfWindows
    def test_source_map(self):
        """
        This test simulates building two files in a folder, and then moving
        each source to a different folder. Then, the debug session is started
        with the corresponding source maps to have breakpoints and frames
        working.
        """
        session = self.build_and_create_session()
        other_basename = "other-copy.c"
        other_path = self.getBuildArtifact(other_basename)

        source_folder = os.path.dirname(self.main_path)

        new_main_folder = os.path.join(source_folder, "moved_main")
        new_other_folder = os.path.join(source_folder, "moved_other")

        new_main_path = os.path.join(new_main_folder, self.main_basename)
        new_other_path = os.path.join(new_other_folder, other_basename)

        # Move the sources.
        os.mkdir(new_main_folder)
        os.mkdir(new_other_folder)
        shutil.move(self.main_path, new_main_path)
        shutil.move(other_path, new_other_path)

        main_line = line_number("main.cpp", "break 12")
        other_line = line_number("other.c", "break other")

        program = self.getBuildArtifact("a.out")
        source_map = [
            (source_folder, new_main_folder),
            (source_folder, new_other_folder),
        ]
        with session.configure(LaunchArgs(program, sourceMap=source_map)) as ctx:
            # Breakpoint in main.cpp.
            response = session.set_source_breakpoints(new_main_path, [main_line])
            self.assertEqual(len(response.body.breakpoints), 1)
            [main_bp] = response.body.breakpoints
            self.assertEqual(main_bp.line, main_line)
            self.assertTrue(main_bp.verified)
            breakpoint_source = self.expect_not_none(main_bp.source)
            self.assertEqual(self.main_basename, breakpoint_source.name)
            self.assertEqual(new_main_path, breakpoint_source.path)

            # 2nd breakpoint, which is from a dynamically loaded library.
            response = session.set_source_breakpoints(new_other_path, [other_line])
            [other_bp] = response.body.breakpoints
            self.assertEqual(other_bp.line, other_line)
            self.assertFalse(other_bp.verified)
            breakpoint_source = self.expect_not_none(other_bp.source)
            self.assertEqual(other_basename, breakpoint_source.name)
            self.assertEqual(new_other_path, breakpoint_source.path)
            other_bp_id = self.expect_not_none(other_bp.id)

        stop_event = session.verify_stopped_on_breakpoint(
            other_bp_id, after=ctx.process_event
        )

        # 2nd breakpoint again, which should be valid at this point.
        response = session.set_source_breakpoints(new_other_path, [other_line])
        [other_bp] = response.body.breakpoints
        self.assertEqual(other_bp.line, other_line)
        self.assertTrue(other_bp.verified)
        breakpoint_source = self.expect_not_none(other_bp.source)
        self.assertEqual(other_basename, breakpoint_source.name)
        self.assertEqual(new_other_path, breakpoint_source.path)

        # Now we check the stack trace making sure that we got mapped source paths.
        frame_ctxs = session.thread_context_from(stop_event).frames()
        frames = [ctx.frame for ctx in frame_ctxs]

        frame0_source = self.expect_not_none(frames[0].source)
        self.assertEqual(frame0_source.name, other_basename)
        self.assertEqual(frame0_source.path, new_other_path)

        frame1_source = self.expect_not_none(frames[1].source)
        self.assertEqual(frame1_source.name, self.main_basename)
        self.assertEqual(frame1_source.path, new_main_path)

        # Clear all breakpoints.
        session.set_source_breakpoints(new_main_path, [])
        session.set_source_breakpoints(new_other_path, [])
        session.continue_to_exit()

    @skipIfWindows
    def test_set_and_clear(self):
        """Tests setting and clearing source file and line breakpoints.
        This packet is a bit tricky on the debug adapter side since there
        is no "clearBreakpoints" packet. Source file and line breakpoints
        are set by sending a "setBreakpoints" packet with a source file
        specified and zero or more source lines. If breakpoints have been
        set in the source file before, any existing breakpoints must remain
        set, and any new breakpoints must be created, and any breakpoints
        that were in previous requests and are not in the current request
        must be removed. This function tests this setting and clearing
        and makes sure things happen correctly. It doesn't test hitting
        breakpoints and the functionality of each breakpoint, like
        'conditions' and 'hitCondition' settings."""
        first_line = line_number("main.cpp", "break 12")
        second_line = line_number("main.cpp", "break 13")
        third_line = line_number("main.cpp", "break 14")
        lines = [first_line, third_line, second_line]

        # Visual Studio Code Debug Adapters have no way to specify the file
        # without launching or attaching to a process, so we must start a
        # process in order to be able to set breakpoints.
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        session.initialize_and_launch(LaunchArgs(program))

        # Set 3 breakpoints and verify that they got set correctly.
        response = session.set_source_breakpoints(self.main_path, lines)
        breakpoints = response.body.breakpoints
        self.assertEqual(
            len(breakpoints),
            len(lines),
            f"expect {len(lines)} source breakpoints",
        )
        line_to_id: Dict[int, int] = {}
        for index, breakpoint in enumerate(breakpoints):
            line = self.expect_not_none(breakpoint.line)
            self.assertEqual(line, lines[index])
            # Store the "id" of the breakpoint that was set for later.
            breakpoint_id = self.expect_not_none(breakpoint.id)
            line_to_id[line] = breakpoint_id
            self.assertTrue(breakpoint.verified, "expect breakpoint verified")

        # There is no breakpoint delete packet, clients just send another
        # setBreakpoints packet with the same source file with fewer lines.
        # Below we remove the second line entry and call the setBreakpoints
        # function again. We want to verify that any breakpoints that were set
        # before still have the same "id". This means we didn't clear the
        # breakpoint and set it again at the same location. We also need to
        # verify that the second line location was actually removed.
        lines.remove(second_line)
        # Set 2 breakpoints and verify that the previous breakpoints that were
        # set above are still set.
        response = session.set_source_breakpoints(self.main_path, lines)
        breakpoints = response.body.breakpoints
        self.assertEqual(
            len(breakpoints), len(lines), f"expect {len(lines)} source breakpoints"
        )
        for expected_line, breakpoint in zip(lines, breakpoints):
            line = self.expect_not_none(breakpoint.line)
            self.assertEqual(line, expected_line)
            # Verify the same breakpoints are still set within LLDB by
            # making sure the breakpoint ID didn't change.
            self.assertEqual(
                line_to_id[line],
                breakpoint.id,
                "verify previous breakpoints stayed the same",
            )
            self.assertTrue(breakpoint.verified, "expect breakpoint still verified")

        # Now get the full list of breakpoints set in the target and verify
        # we have only 2 breakpoints set. The response above could have told
        # us about 2 breakpoints, but we want to make sure we don't have the
        # third one still set in the target
        response = session.send_request(DAPTestGetTargetBreakpointsArgs()).result()
        breakpoints = response.body.breakpoints
        self.assertEqual(
            len(breakpoints),
            len(lines),
            f"expect {len(lines)} source breakpoints",
        )
        for breakpoint in breakpoints:
            line = self.expect_not_none(breakpoint.line)
            # Verify the same breakpoints are still set within LLDB by
            # making sure the breakpoint ID didn't change
            self.assertEqual(
                line_to_id[line],
                breakpoint.id,
                "verify previous breakpoints stayed the same",
            )
            self.assertIn(line, lines, "line expected in lines array")
            self.assertTrue(breakpoint.verified, "expect breakpoint still verified")

        # Now clear all breakpoints for the source file by passing down an
        # empty lines array.
        lines = []
        response = session.set_source_breakpoints(self.main_path, lines)
        breakpoints = response.body.breakpoints
        self.assertEqual(
            len(breakpoints),
            len(lines),
            f"expect {len(lines)} source breakpoints",
        )

        # Verify with the target that all breakpoints have been cleared
        response = session.send_request(DAPTestGetTargetBreakpointsArgs()).result()
        breakpoints = response.body.breakpoints
        self.assertEqual(
            len(breakpoints),
            len(lines),
            f"expect {len(lines)} source breakpoints",
        )

        # Now set a breakpoint again in the same source file and verify it
        # was added.
        lines = [second_line]
        response = session.set_source_breakpoints(self.main_path, lines)
        breakpoints = response.body.breakpoints
        self.assertEqual(
            len(breakpoints),
            len(lines),
            f"expect {len(lines)} source breakpoints",
        )
        for breakpoint in breakpoints:
            line = self.expect_not_none(breakpoint.line)
            self.assertIn(line, lines, "line expected in lines array")
            self.assertTrue(breakpoint.verified, "expect breakpoint still verified")

        # Now get the full list of breakpoints set in the target and verify
        # we have only 2 breakpoints set. The response above could have told
        # us about 2 breakpoints, but we want to make sure we don't have the
        # third one still set in the target.
        response = session.send_request(DAPTestGetTargetBreakpointsArgs()).result()
        breakpoints = response.body.breakpoints
        self.assertEqual(
            len(breakpoints),
            len(lines),
            f"expect {len(lines)} source breakpoints",
        )
        for breakpoint in breakpoints:
            line = self.expect_not_none(breakpoint.line)
            self.assertIn(line, lines, "line expected in lines array")
            self.assertTrue(breakpoint.verified, "expect breakpoint still verified")

    @skipIfWindows
    def test_clear_breakpoints_unset_breakpoints(self):
        """Test clearing breakpoints like test_set_and_clear, but clear
        breakpoints by omitting the breakpoints array instead of sending an
        empty one."""
        lines = [
            line_number("main.cpp", "break 12"),
            line_number("main.cpp", "break 13"),
        ]

        # Visual Studio Code Debug Adapters have no way to specify the file
        # without launching or attaching to a process, so we must start a
        # process in order to be able to set breakpoints.
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        with session.configure(LaunchArgs(program)):
            # Set one breakpoint and verify that it got set correctly.
            response = session.set_source_breakpoints(self.main_path, lines)
            breakpoints = response.body.breakpoints
            self.assertEqual(
                len(breakpoints), len(lines), f"expect {len(lines)} source breakpoints"
            )
            for expected_line, breakpoint in zip(lines, breakpoints):
                line = self.expect_not_none(breakpoint.line)
                self.assertEqual(line, expected_line)
                self.expect_not_none(breakpoint.id)
                self.assertTrue(breakpoint.verified, "expect breakpoint verified")

            # Now clear all breakpoints for the source file by not setting the
            # lines or breakpoints array.
            set_bp_args = SetBreakpointsArgs(Source.create(path=self.main_path))
            response = session.send_request(set_bp_args).result()
            breakpoints = response.body.breakpoints
            self.assertEqual(len(breakpoints), 0, "expect no source breakpoints")

            # Verify with the target that all breakpoints have been cleared.
            response = session.send_request(DAPTestGetTargetBreakpointsArgs()).result()
            breakpoints = response.body.breakpoints
            self.assertEqual(len(breakpoints), 0, "expect no source breakpoints")

        session.verify_process_exited()

    @skipIfWindows
    def test_functionality(self):
        """Tests hitting breakpoints and the functionality of a single
        breakpoint, like 'conditions' and 'hitCondition' settings."""
        loop_line = line_number("main.cpp", "// break loop")

        program = self.getBuildArtifact("a.out")

        session = self.build_and_create_session()
        with session.configure(LaunchArgs(program)) as ctx:
            # Set a breakpoint at the loop line with no condition and no
            # hitCondition.
            [loop_bp] = session.resolve_source_breakpoints(self.main_path, [loop_line])
        process_event = ctx.process_event

        # Verify we hit the breakpoint we just set.
        stop_event = session.verify_stopped_on_breakpoint(loop_bp, after=process_event)

        # Make sure i is zero at first breakpoint.
        thread_ctx = session.thread_context_from(stop_event)
        i_var = thread_ctx.top_frame().locals["i"]
        self.assertEqual(i_var.value_as_int, 0, "i != 0 after hitting breakpoint")

        # Update the condition on our breakpoint.
        [condition_bp] = session.resolve_source_breakpoints(
            self.main_path, [SourceBreakpoint(loop_line, condition="i==4")]
        )
        self.assertEqual(
            loop_bp,
            condition_bp,
            "existing breakpoint should have its condition updated",
        )

        stop_event = session.continue_to_breakpoint(loop_bp)
        thread_ctx = session.thread_context_from(stop_event)
        i_var = thread_ctx.top_frame().locals["i"]
        self.assertEqual(i_var.value_as_int, 4, "i != 4 showing conditional works")

        # Update the hitCondition on our breakpoint.
        [hit_condition_bp] = session.resolve_source_breakpoints(
            self.main_path, [SourceBreakpoint(loop_line, hitCondition="2")]
        )
        self.assertEqual(
            loop_bp,
            hit_condition_bp,
            "existing breakpoint should have its condition updated",
        )

        # Continue with a hitCondition of 2 and expect it to skip 1 value.
        stop_event = session.continue_to_breakpoint(loop_bp)
        i_var = thread_ctx.top_frame().locals["i"]
        self.assertEqual(i_var.value_as_int, 6, "i != 6 showing hitCondition works")

        # Continue after hitting our hitCondition and make sure it only goes
        # up by 1.
        stop_event = session.continue_to_breakpoint(loop_bp)
        i_var = thread_ctx.top_frame().locals["i"]
        self.assertEqual(
            i_var.value_as_int, 7, "i != 7 showing post hitCondition hits every time"
        )

        # Clear breakpoints and exit.
        session.set_source_breakpoints(self.main_path, [])
        session.continue_to_exit()

    @skipIfWindows
    def test_column_breakpoints(self):
        """Test setting multiple breakpoints in the same line at different columns."""
        session = self.build_and_create_session()
        loop_line = line_number("main.cpp", "// break loop")

        program = self.getBuildArtifact("a.out")
        with session.configure(LaunchArgs(program)):
            # Set two breakpoints on the loop line at different columns.
            columns = [13, 39]
            source_bps = [
                SourceBreakpoint(line=loop_line, column=column) for column in columns
            ]
            response = session.set_source_breakpoints(self.main_path, source_bps)

            # Verify the breakpoints were set correctly.
            breakpoints = response.body.breakpoints
            self.assertEqual(
                len(breakpoints),
                len(columns),
                f"expect {len(columns)} source breakpoints",
            )
            breakpoint_ids: list[int] = []
            for column, breakpoint in zip(columns, breakpoints):
                self.assertEqual(breakpoint.line, loop_line)
                self.assertEqual(breakpoint.column, column)
                self.assertTrue(breakpoint.verified, "expect breakpoint verified")
                breakpoint_id = self.expect_not_none(breakpoint.id)
                breakpoint_ids.append(breakpoint_id)

        # Continue to the first breakpoint,
        stop_event = session.verify_stopped_on_breakpoint(
            breakpoint_ids[0], after=response
        )

        # We should have stopped right before the call to `twelve`.
        # Step into and check we are inside `twelve`.
        thread_ctx = session.thread_context_from(stop_event)
        thread_ctx.step_in()
        func_name = thread_ctx.top_frame().name
        self.assertEqual(func_name, "twelve(int)")

        # Continue to the second breakpoint.
        stop_event = session.continue_to_breakpoint(breakpoint_ids[1])

        # We should have stopped right before the call to `fourteen`.
        # Step into and check we are inside `fourteen`.
        thread_ctx.step_in()
        func_name = thread_ctx.top_frame().name
        self.assertEqual(func_name, "a::fourteen(int)")

        session.set_source_breakpoints(self.main_path, [])
        session.continue_to_exit()

    @skipIfWindows
    def test_hit_multiple_breakpoints(self):
        """Test that if we hit multiple breakpoints at the same address, they
        all appear in the stop reason."""
        session = self.build_and_create_session()
        breakpoint_lines = [
            line_number("main.cpp", "// break non-breakpointable line"),
            line_number("main.cpp", "// before loop"),
        ]

        program = self.getBuildArtifact("a.out")
        with session.configure(LaunchArgs(program)) as ctx:
            # Set a pair of breakpoints that will both resolve to the same address.
            breakpoint_ids = session.resolve_source_breakpoints(
                self.main_path, breakpoint_lines
            )

        # Verify we hit both of the breakpoints we just set.
        session.verify_multiple_breakpoints_hit(breakpoint_ids, after=ctx.process_event)

        session.continue_to_exit()
