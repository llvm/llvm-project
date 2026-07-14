"""
Test lldb-dap stackTrace request
"""

import os
from typing import List, NamedTuple

from lldbsuite.test.decorators import skipIfWindows
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap.types import (
    LaunchArgs,
    StackFrame,
    StackFrameFormat,
)
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class _RecurseSource(NamedTuple):
    path: str
    end_line: int
    call_line: int
    invocation_line: int


class TestDAP_stackTrace(DAPTestCaseBase):
    def verify_stack_frames(self, start_idx: int, stack_frames: List[StackFrame]):
        recourse_source = self.recourse_source
        for frame_idx, frame in enumerate(stack_frames, start_idx):
            # Don't care about frames above main.
            if frame_idx > 40:
                return
            self.verify_stack_frame(frame_idx, frame, recourse_source)

    def verify_stack_frame(
        self,
        frame_idx: int,
        stack_frame: StackFrame,
        r_source: _RecurseSource,
    ):
        frame_name = stack_frame.name
        frame_source = self.expect_not_none(stack_frame.source)
        frame_source_path = frame_source.path
        frame_line = stack_frame.line

        if frame_idx == 0:
            expected_line = r_source.end_line
            expected_name = "recurse"
        elif frame_idx < 40:
            expected_line = r_source.call_line
            expected_name = "recurse"
        else:
            expected_line = r_source.invocation_line
            expected_name = "main"

        self.assertEqual(
            frame_name,
            expected_name,
            f'frame #{frame_idx} name "{frame_name}" == "{expected_name}"',
        )
        self.assertEqual(
            frame_source_path,
            r_source.path,
            f'frame #{frame_idx} source "{frame_source_path}" == "{r_source.path}"',
        )
        self.assertEqual(
            frame_line,
            expected_line,
            f"frame #{frame_idx} line {frame_line} == {expected_line}",
        )

    def test_stackTrace(self):
        """Test the 'stackTrace' packet and all its variants."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source_file = self.getSourcePath("main.c")
        self.recourse_source = _RecurseSource(
            path=os.path.realpath(source_file),
            end_line=line_number(source_file, "recurse end"),
            call_line=line_number(source_file, "recurse call"),
            invocation_line=line_number(source_file, "recurse invocation"),
        )

        with session.configure(LaunchArgs(program=program)) as ctx:
            # Set a breakpoint at the point of deepest recursion.
            breakpoint_ids = session.resolve_source_breakpoints(
                source_file, [self.recourse_source.end_line]
            )

        stop_event = session.verify_stopped_on_breakpoint(
            breakpoint_ids, after=ctx.process_event
        )
        thread_id = self.expect_not_none(stop_event.body.threadId)

        start_frame = 0
        # Verify that we get all stack frames with no arguments.
        response = session.stack_trace(thread_id)
        stack_frames = response.body.stackFrames
        total_frames = response.body.totalFrames
        frame_count = len(stack_frames)
        self.assertGreaterEqual(
            frame_count, 40, "verify we get at least 40 frames for all frames"
        )
        self.assertEqual(
            total_frames,
            frame_count,
            "verify total frames returns a speculative page size",
        )
        self.verify_stack_frames(start_frame, stack_frames)

        # Verify that totalFrames includes a speculative page size of additional
        # frames when startFrame=0 and levels=10.
        response = session.stack_trace(thread_id, startFrame=0, levels=10)
        stack_frames = response.body.stackFrames
        total_frames = response.body.totalFrames
        self.assertEqual(len(stack_frames), 10, "verify we get levels=10 frames")

        page_size = 20  # Number of additional frames added by lldb-dap to a paginated stack trace.
        self.assertEqual(
            total_frames,
            len(stack_frames) + page_size,
            "verify total frames returns a speculative page size",
        )
        self.verify_stack_frames(start_frame, stack_frames)

        # Verify all stack frames by specifying startFrame=0 and no levels.
        response = session.stack_trace(thread_id, startFrame=start_frame)
        stack_frames = response.body.stackFrames
        self.assertEqual(
            frame_count,
            len(stack_frames),
            f"verify same number of frames with startFrame={start_frame}",
        )
        self.verify_stack_frames(start_frame, stack_frames)

        # Verify all stack frames by specifying startFrame=0 and levels=0.
        levels = 0
        response = session.stack_trace(thread_id, startFrame=start_frame, levels=levels)
        stack_frames = response.body.stackFrames
        self.assertEqual(
            frame_count,
            len(stack_frames),
            f"verify same number of frames with startFrame={start_frame} and levels={levels}",
        )
        self.verify_stack_frames(start_frame, stack_frames)

        # Get only the first stack frame by specifying startFrame=0 and levels=1.
        levels = 1
        response = session.stack_trace(thread_id, startFrame=start_frame, levels=levels)
        stack_frames = response.body.stackFrames
        self.assertEqual(
            levels,
            len(stack_frames),
            f"verify one frame with {start_frame=} and {levels=}",
        )
        self.verify_stack_frames(start_frame, stack_frames)

        # Get only the first 3 stack frames by specifying startFrame=0 and levels=3.
        levels = 3
        response = session.stack_trace(thread_id, startFrame=start_frame, levels=levels)
        stack_frames = response.body.stackFrames
        self.assertEqual(
            levels,
            len(stack_frames),
            f"verify {levels} frames with {start_frame=} and {levels=}",
        )
        self.verify_stack_frames(start_frame, stack_frames)

        # Get the first 16 stack frames by specifying startFrame=5 and levels=16.
        start_frame = 5
        levels = 16
        response = session.stack_trace(thread_id, startFrame=start_frame, levels=levels)
        stack_frames = response.body.stackFrames
        self.assertEqual(
            levels,
            len(stack_frames),
            f"verify {levels} frames with {start_frame=} and {levels=}",
        )
        self.verify_stack_frames(start_frame, stack_frames)

        # Verify that the count is capped correctly when we ask for too many frames.
        start_frame = 5
        levels = 1000
        response = session.stack_trace(thread_id, startFrame=start_frame, levels=levels)
        stack_frames = response.body.stackFrames
        total_frames = response.body.totalFrames
        self.assertEqual(
            len(stack_frames),
            frame_count - start_frame,
            f"verify fewer than 1000 frames with {start_frame=} and {levels=}",
        )
        self.assertEqual(
            total_frames,
            frame_count,
            "verify we get the correct value for totalFrames count "
            "when requested frames do not start at index 0",
        )
        self.verify_stack_frames(start_frame, stack_frames)

        # Verify that levels=0 works with a non-zero startFrame.
        start_frame = 5
        levels = 0
        response = session.stack_trace(thread_id, startFrame=start_frame, levels=levels)
        stack_frames = response.body.stackFrames
        self.assertEqual(
            len(stack_frames),
            frame_count - start_frame,
            f"verify all remaining frames with startFrame={start_frame} and levels={levels}",
        )
        self.verify_stack_frames(start_frame, stack_frames)

        # Verify that we get no frames when startFrame is too high.
        start_frame = 1000
        levels = 1
        response = session.stack_trace(thread_id, startFrame=start_frame, levels=levels)
        stack_frames = response.body.stackFrames
        self.assertEqual(
            0, len(stack_frames), "verify zero frames with startFrame out of bounds"
        )

    @skipIfWindows
    def test_function_name_with_args(self):
        """Test that a stack frame's name includes its argument values."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        launch_args = LaunchArgs(
            program, customFrameFormat="${function.name-with-args}"
        )
        with session.configure(launch_args) as ctx:
            source = "main.c"
            session.resolve_source_breakpoints(
                source, [line_number(source, "recurse end")]
            )
        stop_event = session.verify_stopped_on_breakpoint(after=ctx.process_event)
        thread_ctx = session.thread_context_from(stop_event)

        frame = thread_ctx.top_frame().frame
        self.assertEqual(frame.name, "recurse(x=1)")

    @skipIfWindows
    def test_stack_frame_format(self):
        """
        Test the StackFrameFormat options.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        with session.configure(LaunchArgs(program)) as ctx:
            source = "main.c"
            bp_line = line_number(source, "recurse end")
            session.resolve_source_breakpoints(source, [bp_line])

        stop_event = session.verify_stopped_on_breakpoint(after=ctx.process_event)
        thread_ctx = session.thread_context_from(stop_event)
        frame = thread_ctx.top_frame(format=StackFrameFormat(parameters=True))
        self.assertEqual(frame.name, "recurse(x=1)")

        frame = thread_ctx.top_frame(format=StackFrameFormat(parameterNames=True))
        self.assertEqual(frame.name, "recurse(x=1)")

        frame = thread_ctx.top_frame(format=StackFrameFormat(parameterValues=True))
        self.assertEqual(frame.name, "recurse(x=1)")

        format = StackFrameFormat(parameters=False, line=True)
        frame = thread_ctx.top_frame(format=format)
        self.assertEqual(frame.name, f"main.c:{bp_line}:5 recurse")

        format = StackFrameFormat(parameters=False, module=True)
        frame = thread_ctx.top_frame(format=format)
        self.assertEqual(frame.name, "a.out recurse")

    @skipIfWindows
    def test_stack_frame_module_id(self) -> None:
        """Test that each stack frame's moduleId matches the loaded module's id."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = self.getSourcePath("main.c")
        lines = [line_number(source, "recurse end")]

        with session.configure(LaunchArgs(program=program)) as ctx:
            breakpoint_ids = session.resolve_source_breakpoints(source, lines)

        stop_event = session.verify_stopped_on_breakpoint(
            breakpoint_ids, after=ctx.process_event
        )
        thread_id = self.expect_not_none(stop_event.body.threadId)
        modules = session.get_modules()
        stack_frames = session.stack_trace(thread_id).body.stackFrames

        for frame in stack_frames:
            module_id = frame.moduleId
            source_name = frame.source and frame.source.name
            if module_id is None or source_name is None:
                continue

            if source_name in modules:
                expected_id = modules[source_name].id
                self.assertEqual(
                    module_id,
                    expected_id,
                    f"expected moduleId '{expected_id}' for {source_name}, got: {module_id}",
                )
