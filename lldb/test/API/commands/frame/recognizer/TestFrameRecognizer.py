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

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    def test_frame_recognizer_1(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Clear internal & plugins recognizers that get initialized at launch
        self.runCmd("frame recognizer clear")

        self.runCmd("command script import " + os.path.join(self.getSourceDir(), "recognizer.py"))

        self.expect("frame recognizer list",
                    substrs=['no matching results found.'])

        self.runCmd("frame recognizer add -l recognizer.MyFrameRecognizer -s a.out -n foo")

        self.expect("frame recognizer list",
                    substrs=['0: recognizer.MyFrameRecognizer, module a.out, symbol foo'])

        self.runCmd("frame recognizer add -l recognizer.MyOtherFrameRecognizer -s a.out -n bar -x")

        self.expect(
            "frame recognizer list",
            substrs=[
                '1: recognizer.MyOtherFrameRecognizer, module a.out, symbol bar (regexp)',
                '0: recognizer.MyFrameRecognizer, module a.out, symbol foo'
            ])

        self.runCmd("frame recognizer delete 0")

        self.expect("frame recognizer list",
                    substrs=['1: recognizer.MyOtherFrameRecognizer, module a.out, symbol bar (regexp)'])

        self.runCmd("frame recognizer clear")

        self.expect("frame recognizer list",
                    substrs=['no matching results found.'])

        self.runCmd("frame recognizer add -l recognizer.MyFrameRecognizer -s a.out -n foo")

        target, process, thread, _ = lldbutil.run_to_name_breakpoint(self, "foo",
                                                                 exe_name = exe)
        frame = thread.GetSelectedFrame()

        self.expect("frame variable",
                    substrs=['(int) a = 42', '(int) b = 56'])

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
        self.assertEqual(variables.GetValueAtIndex(0).GetValueType(), lldb.eValueTypeVariableArgument)
        self.assertEqual(variables.GetValueAtIndex(1).name, "b")
        self.assertEqual(variables.GetValueAtIndex(1).signed, 56)
        self.assertEqual(variables.GetValueAtIndex(1).GetValueType(), lldb.eValueTypeVariableArgument)

        self.expect("frame recognizer info 0",
                    substrs=['frame 0 is recognized by recognizer.MyFrameRecognizer'])

        self.expect("frame recognizer info 999", error=True,
                    substrs=['no frame with index 999'])

        self.expect("frame recognizer info 1",
                    substrs=['frame 1 not recognized by any recognizer'])

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

    @skipUnlessDarwin
    def test_frame_recognizer_multi_symbol(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Clear internal & plugins recognizers that get initialized at launch
        self.runCmd("frame recognizer clear")

        self.runCmd("command script import " + os.path.join(self.getSourceDir(), "recognizer.py"))

        self.expect("frame recognizer list",
                    substrs=['no matching results found.'])

        self.runCmd("frame recognizer add -l recognizer.MyFrameRecognizer -s a.out -n foo -n bar")

        self.expect("frame recognizer list",
                    substrs=['recognizer.MyFrameRecognizer, module a.out, symbol foo, symbol bar'])

        target, process, thread, _ = lldbutil.run_to_name_breakpoint(self, "foo",
                                                                 exe_name = exe)
        frame = thread.GetSelectedFrame()

        self.expect("frame recognizer info 0",
                    substrs=['frame 0 is recognized by recognizer.MyFrameRecognizer'])

        target, process, thread, _ = lldbutil.run_to_name_breakpoint(self, "bar",
                                                                 exe_name = exe)
        frame = thread.GetSelectedFrame()

        self.expect("frame recognizer info 0",
                    substrs=['frame 0 is recognized by recognizer.MyFrameRecognizer'])

    @skipUnlessDarwin
    def test_frame_recognizer_target_specific(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Clear internal & plugins recognizers that get initialized at launch
        self.runCmd("frame recognizer clear")

        # Create a target.
        target, process, thread, _ = lldbutil.run_to_name_breakpoint(self, "foo",
                                                                 exe_name = exe)

        self.runCmd("command script import " + os.path.join(self.getSourceDir(), "recognizer.py"))

        # Check that this doesn't contain our own FrameRecognizer somehow.
        self.expect("frame recognizer list",
                    matching=False, substrs=['MyFrameRecognizer'])

        # Add a frame recognizer in that target.
        self.runCmd("frame recognizer add -l recognizer.MyFrameRecognizer -s a.out -n foo -n bar")

        self.expect("frame recognizer list",
                    substrs=['recognizer.MyFrameRecognizer, module a.out, symbol foo, symbol bar'])

        self.expect("frame recognizer info 0",
                    substrs=['frame 0 is recognized by recognizer.MyFrameRecognizer'])

        # Create a second target. That one shouldn't have the frame recognizer.
        target, process, thread, _ = lldbutil.run_to_name_breakpoint(self, "bar",
                                                                 exe_name = exe)

        self.expect("frame recognizer info 0",
                    substrs=['frame 0 not recognized by any recognizer'])

        # Add a frame recognizer to the new target.
        self.runCmd("frame recognizer add -l recognizer.MyFrameRecognizer -s a.out -n bar")

        self.expect("frame recognizer list",
                    substrs=['recognizer.MyFrameRecognizer, module a.out, symbol bar'])

        # Now the new target should also recognize the frame.
        self.expect("frame recognizer info 0",
                    substrs=['frame 0 is recognized by recognizer.MyFrameRecognizer'])

    @no_debug_info_test
    def test_frame_recognizer_delete_invalid_arg(self):
        self.expect("frame recognizer delete a", error=True,
                    substrs=["error: 'a' is not a valid recognizer id."])
        self.expect("frame recognizer delete \"\"", error=True,
                    substrs=["error: '' is not a valid recognizer id."])
        self.expect("frame recognizer delete -1", error=True,
                    substrs=["error: '-1' is not a valid recognizer id."])
        self.expect("frame recognizer delete 4294967297", error=True,
                    substrs=["error: '4294967297' is not a valid recognizer id."])

    @no_debug_info_test
    def test_frame_recognizer_info_invalid_arg(self):
        self.expect("frame recognizer info a", error=True,
                    substrs=["error: 'a' is not a valid frame index."])
        self.expect("frame recognizer info \"\"", error=True,
                    substrs=["error: '' is not a valid frame index."])
        self.expect("frame recognizer info -1", error=True,
                    substrs=["error: '-1' is not a valid frame index."])
        self.expect("frame recognizer info 4294967297", error=True,
                    substrs=["error: '4294967297' is not a valid frame index."])
