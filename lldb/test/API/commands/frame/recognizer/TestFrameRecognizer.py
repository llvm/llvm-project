# encoding: utf-8
"""
Test lldb's frame recognizers.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

import recognizer


class FrameRecognizerTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_recognizer_1(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target, process, thread, _ = lldbutil.run_to_name_breakpoint(
            self, "foo", exe_name=exe
        )
        frame = thread.GetSelectedFrame()

        # Clear internal & plugins recognizers that get initialized at launch
        self.runCmd("frame recognizer clear")

        self.runCmd(
            "command script import "
            + os.path.join(self.getSourceDir(), "recognizer.py")
        )

        self.expect("frame recognizer list", substrs=["no matching results found."])

        self.runCmd(
            "frame recognizer add -l recognizer.MyFrameRecognizer -s a.out -n foo"
        )

        self.expect(
            "frame recognizer list",
            substrs=[
                "0: recognizer.MyFrameRecognizer, module a.out, demangled symbol foo"
            ],
        )

        self.runCmd(
            "frame recognizer add -l recognizer.MyOtherFrameRecognizer -s a.out -n bar -x"
        )

        self.expect(
            "frame recognizer list",
            substrs=[
                "1: recognizer.MyOtherFrameRecognizer, module a.out, demangled symbol regex bar",
                "0: recognizer.MyFrameRecognizer, module a.out, demangled symbol foo",
            ],
        )

        self.runCmd("frame recognizer delete 0")

        # Test that it deleted the recognizer with id 0.
        self.expect(
            "frame recognizer list",
            substrs=[
                "1: recognizer.MyOtherFrameRecognizer, module a.out, demangled symbol regex bar"
            ],
        )
        self.expect(
            "frame recognizer list", matching=False, substrs=["MyFrameRecognizer"]
        )

        # Test that an invalid index and deleting the same index again
        # is an error and doesn't do any changes.
        self.expect(
            "frame recognizer delete 2",
            error=True,
            substrs=["error: '2' is not a valid recognizer id."],
        )
        self.expect(
            "frame recognizer delete 0",
            error=True,
            substrs=["error: '0' is not a valid recognizer id."],
        )
        # Recognizers should have the same state as above.
        self.expect(
            "frame recognizer list",
            substrs=[
                "1: recognizer.MyOtherFrameRecognizer, module a.out, demangled symbol regex bar"
            ],
        )
        self.expect(
            "frame recognizer list", matching=False, substrs=["MyFrameRecognizer"]
        )

        self.runCmd("frame recognizer clear")

        self.expect("frame recognizer list", substrs=["no matching results found."])

        self.runCmd(
            "frame recognizer add -l recognizer.MyFrameRecognizer -s a.out -n foo"
        )

        self.expect("frame variable", substrs=["(int) a = 42", "(int) b = 56"])

        # Recognized arguments don't show up by default...
        variables = frame.GetVariables(lldb.SBVariablesOptions())
        self.assertEqual(variables.GetSize(), 0)

        # ...unless you set target.display-recognized-arguments to 1...
        self.runCmd("settings set target.display-recognized-arguments 1")
        variables = frame.GetVariables(lldb.SBVariablesOptions())
        self.assertEqual(variables.GetSize(), 2)

        # ...and you can reset it back to 0 to hide them again...
        self.runCmd("settings set target.display-recognized-arguments 0")
        variables = frame.GetVariables(lldb.SBVariablesOptions())
        self.assertEqual(variables.GetSize(), 0)

        # ... or explicitly ask for them with SetIncludeRecognizedArguments(True).
        opts = lldb.SBVariablesOptions()
        opts.SetIncludeRecognizedArguments(True)
        variables = frame.GetVariables(opts)

        self.assertEqual(variables.GetSize(), 2)
        self.assertEqual(variables.GetValueAtIndex(0).name, "a")
        self.assertEqual(variables.GetValueAtIndex(0).signed, 42)
        self.assertEqual(
            variables.GetValueAtIndex(0).GetValueType(), lldb.eValueTypeVariableArgument
        )
        self.assertEqual(variables.GetValueAtIndex(1).name, "b")
        self.assertEqual(variables.GetValueAtIndex(1).signed, 56)
        self.assertEqual(
            variables.GetValueAtIndex(1).GetValueType(), lldb.eValueTypeVariableArgument
        )

        self.expect(
            "frame recognizer info 0",
            substrs=["frame 0 is recognized by recognizer.MyFrameRecognizer"],
        )

        self.expect(
            "frame recognizer info 999", error=True, substrs=["no frame with index 999"]
        )

        self.expect(
            "frame recognizer info 1",
            substrs=["frame 1 not recognized by any recognizer"],
        )

        # FIXME: The following doesn't work yet, but should be fixed.
        """
        target, process, thread, _ = lldbutil.run_to_name_breakpoint(self, "bar",
                                                                 exe_name = exe)
        frame = thread.GetSelectedFrame()

        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        self.expect("frame variable -t",
                    substrs=['(int *) a = '])

        self.expect("frame variable -t *a",
                    substrs=['*a = 78'])
        """

    def test_frame_recognizer_hiding(self):
        self.build()

        target, process, thread, _ = lldbutil.run_to_name_breakpoint(self, "nested")
        frame = thread.GetSelectedFrame()

        # Sanity check.
        self.expect(
            "thread backtrace", patterns=["frame.*nested", "frame.*baz", "frame.*main"]
        )

        self.expect("frame recognizer clear")
        self.expect(
            "command script import "
            + os.path.join(self.getSourceDir(), "recognizer.py")
        )

        self.expect(
            "frame recognizer add -l recognizer.BazFrameRecognizer -f false -s a.out -n baz"
        )

        self.expect(
            "frame recognizer list",
            substrs=["0: recognizer.BazFrameRecognizer"],
        )

        # Now main should be hidden.
        self.expect("thread backtrace", matching=False, patterns=["frame.*baz"])
        self.assertFalse(frame.IsHidden())
        frame = thread.SetSelectedFrame(1)
        self.assertIn("baz", frame.name)
        self.assertTrue(frame.IsHidden())

        # Test StepOut.
        frame = thread.SetSelectedFrame(0)
        thread.StepOut()
        frame = thread.GetSelectedFrame()
        self.assertIn("main", frame.name)

    def test_frame_recognizer_multi_symbol(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Clear internal & plugins recognizers that get initialized at launch
        self.runCmd("frame recognizer clear")

        self.runCmd(
            "command script import "
            + os.path.join(self.getSourceDir(), "recognizer.py")
        )

        self.expect("frame recognizer list", substrs=["no matching results found."])

        self.runCmd(
            "frame recognizer add -l recognizer.MyFrameRecognizer -s a.out -n foo -n bar"
        )

        self.expect(
            "frame recognizer list",
            substrs=[
                "recognizer.MyFrameRecognizer, module a.out, demangled symbol foo, bar"
            ],
        )

        target, process, thread, _ = lldbutil.run_to_name_breakpoint(
            self, "foo", exe_name=exe
        )
        frame = thread.GetSelectedFrame()

        self.expect(
            "frame recognizer info 0",
            substrs=["frame 0 is recognized by recognizer.MyFrameRecognizer"],
        )

        target, process, thread, _ = lldbutil.run_to_name_breakpoint(
            self, "bar", exe_name=exe
        )
        frame = thread.GetSelectedFrame()

        self.expect(
            "frame recognizer info 0",
            substrs=["frame 0 is recognized by recognizer.MyFrameRecognizer"],
        )

    def test_frame_recognizer_target_specific(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Clear internal & plugins recognizers that get initialized at launch
        self.runCmd("frame recognizer clear")

        # Create a target.
        target, process, thread, _ = lldbutil.run_to_name_breakpoint(
            self, "foo", exe_name=exe
        )

        self.runCmd(
            "command script import "
            + os.path.join(self.getSourceDir(), "recognizer.py")
        )

        # Check that this doesn't contain our own FrameRecognizer somehow.
        self.expect(
            "frame recognizer list", matching=False, substrs=["MyFrameRecognizer"]
        )

        # Add a frame recognizer in that target.
        self.runCmd(
            "frame recognizer add -l recognizer.MyFrameRecognizer -s a.out -n foo -n bar"
        )

        self.expect(
            "frame recognizer list",
            substrs=[
                "recognizer.MyFrameRecognizer, module a.out, demangled symbol foo, bar"
            ],
        )

        self.expect(
            "frame recognizer info 0",
            substrs=["frame 0 is recognized by recognizer.MyFrameRecognizer"],
        )

        # Create a second target. That one shouldn't have the frame recognizer.
        target, process, thread, _ = lldbutil.run_to_name_breakpoint(
            self, "bar", exe_name=exe
        )

        self.expect(
            "frame recognizer info 0",
            substrs=["frame 0 not recognized by any recognizer"],
        )

        # Add a frame recognizer to the new target.
        self.runCmd(
            "frame recognizer add -l recognizer.MyFrameRecognizer -s a.out -n bar"
        )

        self.expect(
            "frame recognizer list",
            substrs=[
                "recognizer.MyFrameRecognizer, module a.out, demangled symbol bar"
            ],
        )

        # Now the new target should also recognize the frame.
        self.expect(
            "frame recognizer info 0",
            substrs=["frame 0 is recognized by recognizer.MyFrameRecognizer"],
        )

    def test_frame_recognizer_not_only_first_instruction(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Clear internal & plugins recognizers that get initialized at launch.
        self.runCmd("frame recognizer clear")

        self.runCmd(
            "command script import "
            + os.path.join(self.getSourceDir(), "recognizer.py")
        )

        self.expect("frame recognizer list", substrs=["no matching results found."])

        # Create a target.
        target, process, thread, _ = lldbutil.run_to_name_breakpoint(
            self, "foo", exe_name=exe
        )

        # Move the PC one instruction further.
        self.runCmd("next")

        # Add a frame recognizer in that target.
        self.runCmd(
            "frame recognizer add -l recognizer.MyFrameRecognizer -s a.out -n foo -n bar"
        )

        # It's not applied to foo(), because frame's PC is not at the first instruction of the function.
        self.expect(
            "frame recognizer info 0",
            substrs=["frame 0 not recognized by any recognizer"],
        )

        # Add a frame recognizer with --first-instruction-only=true.
        self.runCmd("frame recognizer clear")

        self.runCmd(
            "frame recognizer add -l recognizer.MyFrameRecognizer -s a.out -n foo -n bar --first-instruction-only=true"
        )

        # It's not applied to foo(), because frame's PC is not at the first instruction of the function.
        self.expect(
            "frame recognizer info 0",
            substrs=["frame 0 not recognized by any recognizer"],
        )

        # Now add a frame recognizer with --first-instruction-only=false.
        self.runCmd("frame recognizer clear")

        self.runCmd(
            "frame recognizer add -l recognizer.MyFrameRecognizer -s a.out -n foo -n bar --first-instruction-only=false"
        )

        # This time it should recognize the frame.
        self.expect(
            "frame recognizer info 0",
            substrs=["frame 0 is recognized by recognizer.MyFrameRecognizer"],
        )

        opts = lldb.SBVariablesOptions()
        opts.SetIncludeRecognizedArguments(True)
        frame = thread.GetSelectedFrame()
        variables = frame.GetVariables(opts)

        self.assertEqual(variables.GetSize(), 2)
        self.assertEqual(variables.GetValueAtIndex(0).name, "a")
        self.assertEqual(variables.GetValueAtIndex(0).signed, 42)
        self.assertEqual(
            variables.GetValueAtIndex(0).GetValueType(), lldb.eValueTypeVariableArgument
        )
        self.assertEqual(variables.GetValueAtIndex(1).name, "b")
        self.assertEqual(variables.GetValueAtIndex(1).signed, 56)
        self.assertEqual(
            variables.GetValueAtIndex(1).GetValueType(), lldb.eValueTypeVariableArgument
        )

    def test_frame_recognizer_disable(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target, process, thread, _ = lldbutil.run_to_name_breakpoint(
            self, "foo", exe_name=exe
        )

        # Clear internal & plugins recognizers that get initialized at launch.
        self.runCmd("frame recognizer clear")

        self.runCmd(
            "command script import "
            + os.path.join(self.getSourceDir(), "recognizer.py")
        )

        # Add a frame recognizer in that target.
        self.runCmd(
            "frame recognizer add -l recognizer.MyFrameRecognizer -s a.out -n foo -n bar"
        )

        # The frame is recognized
        self.expect(
            "frame recognizer info 0",
            substrs=["frame 0 is recognized by recognizer.MyFrameRecognizer"],
        )

        # Disable the recognizer
        self.runCmd("frame recognizer disable 0")

        self.expect(
            "frame recognizer list",
            substrs=[
                "0: [disabled] recognizer.MyFrameRecognizer, module a.out, demangled symbol foo"
            ],
        )

        self.expect(
            "frame recognizer info 0",
            substrs=["frame 0 not recognized by any recognizer"],
        )

        # Re-enable the recognizer
        self.runCmd("frame recognizer enable 0")

        self.expect(
            "frame recognizer list",
            substrs=[
                "0: recognizer.MyFrameRecognizer, module a.out, demangled symbol foo"
            ],
        )

        self.expect(
            "frame recognizer info 0",
            substrs=["frame 0 is recognized by recognizer.MyFrameRecognizer"],
        )

    @no_debug_info_test
    def test_frame_recognizer_delete_invalid_arg(self):
        self.expect(
            "frame recognizer delete a",
            error=True,
            substrs=["error: 'a' is not a valid recognizer id."],
        )
        self.expect(
            'frame recognizer delete ""',
            error=True,
            substrs=["error: '' is not a valid recognizer id."],
        )
        self.expect(
            "frame recognizer delete -1",
            error=True,
            substrs=["error: '-1' is not a valid recognizer id."],
        )
        self.expect(
            "frame recognizer delete 4294967297",
            error=True,
            substrs=["error: '4294967297' is not a valid recognizer id."],
        )

    @no_debug_info_test
    def test_frame_recognizer_info_invalid_arg(self):
        self.expect(
            "frame recognizer info a",
            error=True,
            substrs=["error: 'a' is not a valid frame index."],
        )
        self.expect(
            'frame recognizer info ""',
            error=True,
            substrs=["error: '' is not a valid frame index."],
        )
        self.expect(
            "frame recognizer info -1",
            error=True,
            substrs=["error: '-1' is not a valid frame index."],
        )
        self.expect(
            "frame recognizer info 4294967297",
            error=True,
            substrs=["error: '4294967297' is not a valid frame index."],
        )

    @no_debug_info_test
    def test_frame_recognizer_add_invalid_arg(self):
        self.expect(
            "frame recognizer add -f",
            error=True,
            substrs=["error: last option requires an argument"],
        )
        self.expect(
            "frame recognizer add -f -1",
            error=True,
            substrs=["error: invalid boolean value '-1' passed for -f option"],
        )
        self.expect(
            "frame recognizer add -f foo",
            error=True,
            substrs=["error: invalid boolean value 'foo' passed for -f option"],
        )
