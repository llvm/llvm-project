"""
Test lldb-dap stackTrace request with an extended backtrace thread.
"""


import os

import lldbdap_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbplatformutil import *


class TestDAP_extendedStackTrace(lldbdap_testcase.DAPTestCaseBase):
    @skipUnlessDarwin
    def test_stackTrace(self):
        """
        Tests the 'stackTrace' packet on a thread with an extended backtrace.
        """
        backtrace_recording_lib = findBacktraceRecordingDylib()
        if not backtrace_recording_lib:
            self.skipTest(
                "Skipped because libBacktraceRecording.dylib was present on the system."
            )

        if not os.path.isfile("/usr/lib/system/introspection/libdispatch.dylib"):
            self.skipTest(
                "Skipped because introspection libdispatch dylib is not present."
            )

        program = self.getBuildArtifact("a.out")

        self.build_and_launch(
            program,
            env=[
                "DYLD_LIBRARY_PATH=/usr/lib/system/introspection",
                "DYLD_INSERT_LIBRARIES=" + backtrace_recording_lib,
            ],
            displayExtendedBacktrace=True,
        )
        source = "main.m"
        breakpoint = line_number(source, "breakpoint 1")
        lines = [breakpoint]

        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )

        events = self.continue_to_next_stop()

        stackFrames, totalFrames = self.get_stackFrames_and_totalFramesCount(
            threadId=events[0]["body"]["threadId"]
        )
        self.assertGreaterEqual(len(stackFrames), 3, "expect >= 3 frames")
        self.assertEqual(len(stackFrames), totalFrames)
        self.assertEqual(stackFrames[0]["name"], "one")
        self.assertEqual(stackFrames[1]["name"], "two")
        self.assertEqual(stackFrames[2]["name"], "three")

        stackLabels = [
            (i, frame)
            for i, frame in enumerate(stackFrames)
            if frame.get("presentationHint", "") == "label"
        ]
        self.assertEqual(len(stackLabels), 2, "expected two label stack frames")
        self.assertRegex(
            stackLabels[0][1]["name"],
            "Enqueued from com.apple.root.default-qos \(Thread \d\)",
        )
        self.assertRegex(
            stackLabels[1][1]["name"],
            "Enqueued from com.apple.main-thread \(Thread \d\)",
        )

        for i, frame in stackLabels:
            # Ensure requesting startFrame+levels across thread backtraces works as expected.
            stackFrames, totalFrames = self.get_stackFrames_and_totalFramesCount(
                threadId=events[0]["body"]["threadId"], startFrame=i - 1, levels=3
            )
            self.assertEqual(len(stackFrames), 3, "expected 3 frames with levels=3")
            self.assertGreaterEqual(
                totalFrames, i + 3, "total frames should include a pagination offset"
            )
            self.assertEqual(stackFrames[1], frame)

            # Ensure requesting startFrame+levels at the beginning of a thread backtraces works as expected.
            stackFrames, totalFrames = self.get_stackFrames_and_totalFramesCount(
                threadId=events[0]["body"]["threadId"], startFrame=i, levels=3
            )
            self.assertEqual(len(stackFrames), 3, "expected 3 frames with levels=3")
            self.assertGreaterEqual(
                totalFrames, i + 3, "total frames should include a pagination offset"
            )
            self.assertEqual(stackFrames[0], frame)

            # Ensure requests with startFrame+levels that end precisely on the last frame includes the totalFrames pagination offset.
            stackFrames, totalFrames = self.get_stackFrames_and_totalFramesCount(
                threadId=events[0]["body"]["threadId"], startFrame=i - 1, levels=1
            )
            self.assertEqual(len(stackFrames), 1, "expected 1 frames with levels=1")
            self.assertGreaterEqual(
                totalFrames, i, "total frames should include a pagination offset"
            )
