"""
Test lldb-vscode setBreakpoints request
"""


import unittest2
import vscode
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase
import os


class TestVSCode_stackTrace(lldbvscode_testcase.VSCodeTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)
    name_key_path = ['name']
    source_key_path = ['source', 'path']
    line_key_path = ['line']

    def verify_stackFrames(self, start_idx, stackFrames):
        frame_idx = start_idx
        for stackFrame in stackFrames:
            # Don't care about frame above main
            if frame_idx > 20:
                return
            self.verify_stackFrame(frame_idx, stackFrame)
            frame_idx += 1

    def verify_stackFrame(self, frame_idx, stackFrame):
        frame_name = self.get_dict_value(stackFrame, self.name_key_path)
        frame_source = self.get_dict_value(stackFrame, self.source_key_path)
        frame_line = self.get_dict_value(stackFrame, self.line_key_path)
        if frame_idx == 0:
            expected_line = self.recurse_end
            expected_name = 'recurse'
        elif frame_idx < 20:
            expected_line = self.recurse_call
            expected_name = 'recurse'
        else:
            expected_line = self.recurse_invocation
            expected_name = 'main'
        self.assertEquals(frame_name, expected_name,
                        'frame #%i name "%s" == "%s"' % (
                            frame_idx, frame_name, expected_name))
        self.assertEquals(frame_source, self.source_path,
                        'frame #%i source "%s" == "%s"' % (
                            frame_idx, frame_source, self.source_path))
        self.assertEquals(frame_line, expected_line,
                        'frame #%i line %i == %i' % (frame_idx, frame_line,
                                                     expected_line))

    @skipIfWindows
    @skipIfRemote
    def test_stackTrace(self):
        '''
            Tests the 'stackTrace' packet and all its variants.
        '''
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = 'main.c'
        self.source_path = os.path.join(os.getcwd(), source)
        self.recurse_end = line_number(source, 'recurse end')
        self.recurse_call = line_number(source, 'recurse call')
        self.recurse_invocation = line_number(source, 'recurse invocation')

        lines = [self.recurse_end]

        # Set breakpoint at a point of deepest recuusion
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEquals(len(breakpoint_ids), len(lines),
                        "expect correct number of breakpoints")

        self.continue_to_breakpoints(breakpoint_ids)
        startFrame = 0
        # Verify we get all stack frames with no arguments
        (stackFrames, totalFrames) = self.get_stackFrames_and_totalFramesCount()
        frameCount = len(stackFrames)
        self.assertTrue(frameCount >= 20,
                        'verify we get at least 20 frames for all frames')
        self.assertEquals(totalFrames, frameCount,
                        'verify we get correct value for totalFrames count')
        self.verify_stackFrames(startFrame, stackFrames)

        # Verify all stack frames by specifying startFrame = 0 and levels not
        # specified
        stackFrames = self.get_stackFrames(startFrame=startFrame)
        self.assertEquals(frameCount, len(stackFrames),
                        ('verify same number of frames with startFrame=%i') % (
                            startFrame))
        self.verify_stackFrames(startFrame, stackFrames)

        # Verify all stack frames by specifying startFrame = 0 and levels = 0
        levels = 0
        stackFrames = self.get_stackFrames(startFrame=startFrame,
                                           levels=levels)
        self.assertEquals(frameCount, len(stackFrames),
                        ('verify same number of frames with startFrame=%i and'
                         ' levels=%i') % (startFrame, levels))
        self.verify_stackFrames(startFrame, stackFrames)

        # Get only the first stack frame by sepcifying startFrame = 0 and
        # levels = 1
        levels = 1
        stackFrames = self.get_stackFrames(startFrame=startFrame,
                                           levels=levels)
        self.assertEquals(levels, len(stackFrames),
                        ('verify one frame with startFrame=%i and'
                         ' levels=%i') % (startFrame, levels))
        self.verify_stackFrames(startFrame, stackFrames)

        # Get only the first 3 stack frames by sepcifying startFrame = 0 and
        # levels = 3
        levels = 3
        stackFrames = self.get_stackFrames(startFrame=startFrame,
                                           levels=levels)
        self.assertEquals(levels, len(stackFrames),
                        ('verify %i frames with startFrame=%i and'
                         ' levels=%i') % (levels, startFrame, levels))
        self.verify_stackFrames(startFrame, stackFrames)

        # Get only the first 15 stack frames by sepcifying startFrame = 5 and
        # levels = 16
        startFrame = 5
        levels = 16
        stackFrames = self.get_stackFrames(startFrame=startFrame,
                                           levels=levels)
        self.assertEquals(levels, len(stackFrames),
                        ('verify %i frames with startFrame=%i and'
                         ' levels=%i') % (levels, startFrame, levels))
        self.verify_stackFrames(startFrame, stackFrames)

        # Verify we cap things correctly when we ask for too many frames
        startFrame = 5
        levels = 1000
        (stackFrames, totalFrames) = self.get_stackFrames_and_totalFramesCount(
                                            startFrame=startFrame,
                                            levels=levels)
        self.assertEquals(len(stackFrames), frameCount - startFrame,
                        ('verify less than 1000 frames with startFrame=%i and'
                         ' levels=%i') % (startFrame, levels))
        self.assertEquals(totalFrames, frameCount,
                        'verify we get correct value for totalFrames count '
                        'when requested frames not from 0 index')
        self.verify_stackFrames(startFrame, stackFrames)

        # Verify level=0 works with non-zerp start frame
        startFrame = 5
        levels = 0
        stackFrames = self.get_stackFrames(startFrame=startFrame,
                                           levels=levels)
        self.assertEquals(len(stackFrames), frameCount - startFrame,
                        ('verify less than 1000 frames with startFrame=%i and'
                         ' levels=%i') % (startFrame, levels))
        self.verify_stackFrames(startFrame, stackFrames)

        # Verify we get not frames when startFrame is too high
        startFrame = 1000
        levels = 1
        stackFrames = self.get_stackFrames(startFrame=startFrame,
                                           levels=levels)
        self.assertEquals(0, len(stackFrames),
                        'verify zero frames with startFrame out of bounds')
