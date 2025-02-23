"""
Test lldb-dap stackTrace request
"""


import os

import lldbdap_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestDAP_stackTrace(lldbdap_testcase.DAPTestCaseBase):
    name_key_path = ["name"]
    source_key_path = ["source", "path"]
    line_key_path = ["line"]

    # stackTrace additioanl frames for paginated traces
    page_size = 20

    def verify_stackFrames(self, start_idx, stackFrames):
        frame_idx = start_idx
        for stackFrame in stackFrames:
            # Don't care about frame above main
            if frame_idx > 40:
                return
            self.verify_stackFrame(frame_idx, stackFrame)
            frame_idx += 1

    def verify_stackFrame(self, frame_idx, stackFrame):
        frame_name = self.get_dict_value(stackFrame, self.name_key_path)
        frame_source = self.get_dict_value(stackFrame, self.source_key_path)
        frame_line = self.get_dict_value(stackFrame, self.line_key_path)
        if frame_idx == 0:
            expected_line = self.recurse_end
            expected_name = "recurse"
        elif frame_idx < 40:
            expected_line = self.recurse_call
            expected_name = "recurse"
        else:
            expected_line = self.recurse_invocation
            expected_name = "main"
        self.assertEqual(
            frame_name,
            expected_name,
            'frame #%i name "%s" == "%s"' % (frame_idx, frame_name, expected_name),
        )
        self.assertEqual(
            frame_source,
            self.source_path,
            'frame #%i source "%s" == "%s"'
            % (frame_idx, frame_source, self.source_path),
        )
        self.assertEqual(
            frame_line,
            expected_line,
            "frame #%i line %i == %i" % (frame_idx, frame_line, expected_line),
        )

    @skipIfWindows
    def test_stackTrace(self):
        """
        Tests the 'stackTrace' packet and all its variants.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.c"
        self.source_path = os.path.join(os.getcwd(), source)
        self.recurse_end = line_number(source, "recurse end")
        self.recurse_call = line_number(source, "recurse call")
        self.recurse_invocation = line_number(source, "recurse invocation")
        self.qsort_call = line_number(source, "qsort call")

        lines = [self.recurse_end, self.qsort_call]

        # Set breakpoint at a point of deepest recuusion
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )

        self.continue_to_breakpoints(breakpoint_ids)
        startFrame = 0
        # Verify we get all stack frames with no arguments
        (stackFrames, totalFrames) = self.get_stackFrames_and_totalFramesCount()
        frameCount = len(stackFrames)
        self.assertGreaterEqual(
            frameCount, 40, "verify we get at least 40 frames for all frames"
        )
        self.assertEqual(
            totalFrames,
            frameCount,
            "verify total frames returns a speculative page size",
        )
        self.verify_stackFrames(startFrame, stackFrames)

        # Verify totalFrames contains a speculative page size of additional frames with startFrame = 0 and levels = 0
        (stackFrames, totalFrames) = self.get_stackFrames_and_totalFramesCount(
            startFrame=0, levels=10
        )
        self.assertEqual(len(stackFrames), 10, "verify we get levels=10 frames")
        self.assertEqual(
            totalFrames,
            len(stackFrames) + self.page_size,
            "verify total frames returns a speculative page size",
        )
        self.verify_stackFrames(startFrame, stackFrames)

        # Verify all stack frames by specifying startFrame = 0 and levels not
        # specified
        stackFrames = self.get_stackFrames(startFrame=startFrame)
        self.assertEqual(
            frameCount,
            len(stackFrames),
            ("verify same number of frames with startFrame=%i") % (startFrame),
        )
        self.verify_stackFrames(startFrame, stackFrames)

        # Verify all stack frames by specifying startFrame = 0 and levels = 0
        levels = 0
        stackFrames = self.get_stackFrames(startFrame=startFrame, levels=levels)
        self.assertEqual(
            frameCount,
            len(stackFrames),
            ("verify same number of frames with startFrame=%i and" " levels=%i")
            % (startFrame, levels),
        )
        self.verify_stackFrames(startFrame, stackFrames)

        # Get only the first stack frame by sepcifying startFrame = 0 and
        # levels = 1
        levels = 1
        stackFrames = self.get_stackFrames(startFrame=startFrame, levels=levels)
        self.assertEqual(
            levels,
            len(stackFrames),
            ("verify one frame with startFrame=%i and" " levels=%i")
            % (startFrame, levels),
        )
        self.verify_stackFrames(startFrame, stackFrames)

        # Get only the first 3 stack frames by sepcifying startFrame = 0 and
        # levels = 3
        levels = 3
        stackFrames = self.get_stackFrames(startFrame=startFrame, levels=levels)
        self.assertEqual(
            levels,
            len(stackFrames),
            ("verify %i frames with startFrame=%i and" " levels=%i")
            % (levels, startFrame, levels),
        )
        self.verify_stackFrames(startFrame, stackFrames)

        # Get only the first 15 stack frames by sepcifying startFrame = 5 and
        # levels = 16
        startFrame = 5
        levels = 16
        stackFrames = self.get_stackFrames(startFrame=startFrame, levels=levels)
        self.assertEqual(
            levels,
            len(stackFrames),
            ("verify %i frames with startFrame=%i and" " levels=%i")
            % (levels, startFrame, levels),
        )
        self.verify_stackFrames(startFrame, stackFrames)

        # Verify we cap things correctly when we ask for too many frames
        startFrame = 5
        levels = 1000
        (stackFrames, totalFrames) = self.get_stackFrames_and_totalFramesCount(
            startFrame=startFrame, levels=levels
        )
        self.assertEqual(
            len(stackFrames),
            frameCount - startFrame,
            ("verify less than 1000 frames with startFrame=%i and" " levels=%i")
            % (startFrame, levels),
        )
        self.assertEqual(
            totalFrames,
            frameCount,
            "verify we get correct value for totalFrames count "
            "when requested frames not from 0 index",
        )
        self.verify_stackFrames(startFrame, stackFrames)

        # Verify level=0 works with non-zerp start frame
        startFrame = 5
        levels = 0
        stackFrames = self.get_stackFrames(startFrame=startFrame, levels=levels)
        self.assertEqual(
            len(stackFrames),
            frameCount - startFrame,
            ("verify less than 1000 frames with startFrame=%i and" " levels=%i")
            % (startFrame, levels),
        )
        self.verify_stackFrames(startFrame, stackFrames)

        # Verify we do not recive frames when startFrame is out of range
        startFrame = 1000
        levels = 1
        stackFrames = self.get_stackFrames(startFrame=startFrame, levels=levels)
        self.assertEqual(
            0, len(stackFrames), "verify zero frames with startFrame out of bounds"
        )

        # Verify a stack frame from an external library (libc`qsort) to ensure
        # frames without source code return a valid source reference.
        self.continue_to_breakpoints(breakpoint_ids)
        (stackFrames, totalFrames) = self.get_stackFrames_and_totalFramesCount()
        frameCount = len(stackFrames)
        self.assertGreaterEqual(
            frameCount, 3, "verify we get frames from system librarys (libc qsort)"
        )
        self.assertEqual(
            totalFrames,
            frameCount,
            "verify total frames returns a speculative page size",
        )

        frame = stackFrames.pop(0)
        frame_name = self.get_dict_value(frame, ["name"])
        self.assertRegex(frame_name, 'comp')
        self.assertEqual(self.get_dict_value(frame, ['line']), 14)
        self.assertNotIn('sourceReference', frame['source'])

        # libc`qsort may not be the first frame below comp, search upwards
        found_qsort = False
        for frame in stackFrames:
            if 'qsort' not in frame['name']:
                continue
            found_qsort = True
            self.assertIn("sourceReference", frame["source"])
        self.assertTrue(found_qsort, 'verify we found the qsort frame')

    @skipIfWindows
    def test_functionNameWithArgs(self):
        """
        Test that the stack frame without a function name is given its pc in the response.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, customFrameFormat="${function.name-with-args}")
        source = "main.c"

        self.set_source_breakpoints(source, [line_number(source, "recurse end")])

        self.continue_to_next_stop()
        frame = self.get_stackFrames()[0]
        self.assertEqual(frame["name"], "recurse(x=1)")
