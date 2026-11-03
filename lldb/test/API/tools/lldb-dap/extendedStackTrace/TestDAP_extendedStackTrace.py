"""
Test lldb-dap stackTrace request with an extended backtrace thread.
"""

import os

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbplatformutil import findBacktraceRecordingDylib
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs, StackFrameFormat


class TestDAP_extendedStackTrace(DAPTestCaseBase):
    def build_and_run_to_breakpoint(self, display_extended_backtrace: bool = True):
        backtrace_recording_lib = findBacktraceRecordingDylib()
        if not backtrace_recording_lib:
            self.skipTest(
                "Skipped because libBacktraceRecording.dylib was not present on the system."
            )
        if not os.path.isfile("/usr/lib/system/introspection/libdispatch.dylib"):
            self.skipTest(
                "Skipped because introspection libdispatch dylib is not present."
            )

        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = self.getSourcePath("main.m")
        bp_line = line_number(source, "breakpoint 1")

        launch_args = LaunchArgs(
            program=program,
            env=[
                "DYLD_LIBRARY_PATH=/usr/lib/system/introspection",
                f"DYLD_INSERT_LIBRARIES={backtrace_recording_lib}",
            ],
            displayExtendedBacktrace=display_extended_backtrace,
        )
        with session.configure(launch_args) as cm:
            [bp_id] = session.resolve_source_breakpoints(source, [bp_line])

        stop_event = session.verify_stopped_on_breakpoint(bp_id, after=cm.process_event)
        return session, stop_event

    @requireDarwin
    def test_stackTrace(self):
        """Tests the 'stackTrace' packet on a thread with an extended backtrace."""
        session, stop_event = self.build_and_run_to_breakpoint()
        thread_id = self.expect_not_none(stop_event.body.threadId)

        response = session.stack_trace(thread_id)
        stack_frames = response.body.stackFrames
        total_frames = response.body.totalFrames

        self.assertGreaterEqual(len(stack_frames), 3, "expect >= 3 frames")
        self.assertEqual(len(stack_frames), total_frames)
        self.assertEqual(stack_frames[0].name, "one")
        self.assertEqual(stack_frames[1].name, "two")
        self.assertEqual(stack_frames[2].name, "three")

        stack_labels = [
            (i, frame)
            for i, frame in enumerate(stack_frames)
            if frame.presentationHint == "label"
        ]
        self.assertEqual(len(stack_labels), 2, "expected two label stack frames")
        self.assertRegex(
            stack_labels[0][1].name,
            r"Enqueued from com.apple.root.default-qos \(Thread \d\)",
        )
        self.assertRegex(
            stack_labels[1][1].name,
            r"Enqueued from com.apple.main-thread \(Thread \d\)",
        )

        for i, frame in stack_labels:
            # Ensure requesting startFrame+levels across thread backtraces works as expected.
            response = session.stack_trace(thread_id, startFrame=i - 1, levels=3)
            stack_frames = response.body.stackFrames
            total_frames = self.expect_not_none(response.body.totalFrames)
            self.assertEqual(len(stack_frames), 3, "expected 3 frames with levels=3")
            self.assertGreaterEqual(
                total_frames, i + 3, "total frames should include a pagination offset"
            )
            self.assertEqual(stack_frames[1], frame)

            # Ensure requesting startFrame+levels at the beginning of a thread backtrace works as expected.
            response = session.stack_trace(thread_id, startFrame=i, levels=3)
            stack_frames = response.body.stackFrames
            total_frames = self.expect_not_none(response.body.totalFrames)
            self.assertEqual(len(stack_frames), 3, "expected 3 frames with levels=3")
            self.assertGreaterEqual(
                total_frames, i + 3, "total frames should include a pagination offset"
            )
            self.assertEqual(stack_frames[0], frame)

            # Ensure requests with startFrame+levels that end precisely on the
            # last frame include the totalFrames pagination offset.
            response = session.stack_trace(thread_id, startFrame=i - 1, levels=1)
            stack_frames = response.body.stackFrames
            total_frames = self.expect_not_none(response.body.totalFrames)
            self.assertEqual(len(stack_frames), 1, "expected 1 frame with levels=1")
            self.assertGreaterEqual(
                total_frames, i, "total frames should include a pagination offset"
            )

    @requireDarwin
    def test_stackTraceWithFormat(self):
        """Tests the 'stackTrace' packet using stack trace formats."""
        session, stop_event = self.build_and_run_to_breakpoint(
            display_extended_backtrace=False
        )
        thread_id = self.expect_not_none(stop_event.body.threadId)

        response = session.stack_trace(
            thread_id, format=StackFrameFormat(includeAll=True)
        )

        stack_labels = [
            frame
            for frame in response.body.stackFrames
            if frame.presentationHint == "label"
        ]
        self.assertEqual(len(stack_labels), 2, "expected two label stack frames")
