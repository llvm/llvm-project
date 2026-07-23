"""
Test lldb-dap setBreakpoints request
"""

from lldbsuite.test.decorators import skipIfWindows
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_step(DAPTestCaseBase):
    @skipIfWindows
    def test_step(self):
        """
        Tests the stepping in/out/over in threads.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = self.getSourcePath("main.cpp")
        breakpoint1_line = line_number(source, "// breakpoint 1")
        lines = [breakpoint1_line]

        # Set breakpoint in the thread function so we can step the threads.
        with session.configure(LaunchArgs(program)) as ctx:
            session.resolve_source_breakpoints(source, lines)
        process_event = ctx.process_event
        stop_event = session.verify_stopped_on_breakpoint(after=process_event)

        # We have a thread that is stopped at our breakpoint.
        # Get the value of "x" and get the source file and line.
        # These will help us determine if we are stepping
        # correctly. If we step a thread correctly we will verify
        # the correct value for x as it progresses through the
        # program.
        thread = session.thread_context_from(stop_event)
        top_frame = thread.top_frame()
        x1 = top_frame.locals["x"].value_as_int
        src1, line1 = top_frame.source_and_line()

        # Now step into the "recurse()" function call again and
        # verify, using the new value of "x" and the source file
        # and line if we stepped correctly.
        thread.step_in()
        top_frame = thread.top_frame()
        x2 = top_frame.locals["x"].value_as_int
        src2, line2 = top_frame.source_and_line()

        self.assertEqual(x1, x2 + 1, "verify step in variable")
        self.assertLess(line2, line1, "verify step in line")
        self.assertEqual(src1, src2, "verify step in source")

        # Now step out and verify.
        thread.step_out()
        top_frame = thread.top_frame()
        x3 = top_frame.locals["x"].value_as_int
        (src3, line3) = top_frame.source_and_line()
        self.assertEqual(x1, x3, "verify step out variable")
        self.assertGreaterEqual(line3, line1, "verify step out line")
        self.assertEqual(src1, src3, "verify step in source")

        # Step over and verify.
        thread.step_over()
        top_frame = thread.top_frame()
        x4 = top_frame.locals["x"].value_as_int
        (src4, line4) = top_frame.source_and_line()
        self.assertEqual(x4, x3, "verify step over variable")
        self.assertGreater(line4, line3, "verify step over line")
        self.assertEqual(src1, src4, "verify step over source")

        # Step a single assembly instruction.
        # Unfortunately, there is no portable way to verify the correct
        # stepping behavior here, because the generated assembly code
        # depends highly on the compiler, its version, the operating
        # system, and many more factors.
        thread.step_over(granularity="instruction")
        thread.step_in(granularity="instruction")

    def test_step_over_inlined_function(self):
        """
        Test stepping over when the program counter is in another file.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = "main.cpp"

        breakpoint_lines = [line_number(source, "// breakpoint 2")]
        step_over_pos = line_number(source, "// position_after_step_over")
        with session.configure(LaunchArgs(program)) as ctx:
            session.resolve_source_breakpoints(source, breakpoint_lines)
        process_event = ctx.process_event
        stop_event = session.verify_stopped_on_breakpoint(after=process_event)

        thread_ctx = session.thread_context_from(stop_event)
        thread_ctx.step_over()
        levels = 1
        frames = thread_ctx.frames(startFrame=0, levels=levels)
        self.assertEqual(len(frames), levels, "expect current number of frame levels.")

        top_frame = frames[0].frame
        top_frame_source = self.expect_not_none(top_frame.source)
        self.assertEqual(
            top_frame_source.name, source, "expect we are in the same file."
        )
        top_frame_path = self.expect_not_none(top_frame_source.path)
        self.assertTrue(
            top_frame_path.endswith(source),
            f"expect path ending with '{source}'.",
        )
        self.assertEqual(
            top_frame.line,
            step_over_pos,
            f"expect step_over on line {step_over_pos}",
        )

        session.continue_to_exit()
