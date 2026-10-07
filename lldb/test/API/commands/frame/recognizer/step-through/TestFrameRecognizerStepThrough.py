# encoding: utf-8
"""
Test lldb's frame recognizers.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

import recognizer


class TestFrameRecognizerStepThrough(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_recognizer_step_through(self):
        """Test that the step through recognizer works"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Clear internal & plugins recognizers that get initialized at launch
        self.runCmd("frame recognizer clear")

        # Create a target.
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "Stop here to step through", lldb.SBFileSpec("main.c")
        )

        self.runCmd(
            "command script import "
            + os.path.join(self.getSourceDir(), "recognizer.py")
        )

        # Check that this doesn't contain our own FrameRecognizer somehow.
        self.expect(
            "frame recognizer list", matching=False, substrs=["NestedFrameRecognizer"]
        )

        # Add a frame recognizer in that target.
        self.runCmd(
            "frame recognizer add -f 1 -l recognizer.NestedFrameRecognizer -s a.out -n baz"
        )

        self.expect(
            "frame recognizer list",
            substrs=[
                "recognizer.NestedFrameRecognizer, module a.out, demangled symbol baz"
            ],
        )

        # Now do a step in, the step through should kick in and take us to bar.
        thread.StepInto()
        self.assertEqual(thread.frames[0].name, "bar", "Did stop at bar")
        self.assertIn("step in", thread.stop_description, "Reason was correct.")
