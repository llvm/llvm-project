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
        )
        source = "main.m"
        breakpoint = line_number(source, "breakpoint 1")
        lines = [breakpoint]

        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )

        events = self.continue_to_next_stop()
        print("huh", events)
        stackFrames = self.get_stackFrames(threadId=events[0]["body"]["threadId"])
        self.assertGreaterEqual(len(stackFrames), 3, "expect >= 3 frames")
        self.assertEqual(stackFrames[0]["name"], "one")
        self.assertEqual(stackFrames[1]["name"], "two")
        self.assertEqual(stackFrames[2]["name"], "three")

        stackLabels = [
            frame
            for frame in stackFrames
            if frame.get("presentationHint", "") == "label"
        ]
        self.assertEqual(len(stackLabels), 2, "expected two label stack frames")
        self.assertRegex(
            stackLabels[0]["name"],
            "Enqueued from com.apple.root.default-qos \(Thread \d\)",
        )
        self.assertRegex(
            stackLabels[1]["name"], "Enqueued from com.apple.main-thread \(Thread \d\)"
        )
