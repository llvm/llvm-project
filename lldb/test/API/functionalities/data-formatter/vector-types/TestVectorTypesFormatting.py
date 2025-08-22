"""
Check that vector types format properly
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class VectorTypesFormattingTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number("main.cpp", "// break here")

    def setup_and_run_to_breakpoint(self):
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            pass

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

    def vector_formatting_test(self, allow_jit=True):
        v = self.frame().FindVariable("v")
        v.SetPreferSyntheticValue(True)
        v.SetFormat(lldb.eFormatVectorOfFloat32)

        if self.TraceOn():
            print(v)

        self.assertEqual(v.GetNumChildren(), 4, "v as float32[] has 4 children")
        self.assertEqual(
            v.GetChildAtIndex(0).GetData().float[0], 1.25, "child 0 == 1.25"
        )
        self.assertEqual(
            v.GetChildAtIndex(1).GetData().float[0], 1.25, "child 1 == 1.25"
        )
        self.assertEqual(
            v.GetChildAtIndex(2).GetData().float[0], 2.50, "child 2 == 2.50"
        )
        self.assertEqual(
            v.GetChildAtIndex(3).GetData().float[0], 2.50, "child 3 == 2.50"
        )

        jit_flag = "" if allow_jit else " --allow-jit false"
        self.expect(
            f"expr{jit_flag} -f int16_t[] -- v",
            substrs=["(0, 16288, 0, 16288, 0, 16416, 0, 16416)"],
        )
        self.expect(
            f"expr{jit_flag} -f uint128_t[] -- v",
            substrs=["(85236745249553456609335044694184296448)"],
        )
        self.expect(
            f"expr{jit_flag} -f float32[] -- v", substrs=["(1.25, 1.25, 2.5, 2.5)"]
        )

        self.expect(f"expr{jit_flag} -- f4", substrs=["(1.25, 1.25, 2.5, 2.5)"])

        self.expect(f"expr{jit_flag} -- float4(0)", substrs=["(0, 0, 0, 0)"])
        self.expect(f"expr{jit_flag} -- float4(1)", substrs=["(1, 1, 1, 1)"])
        self.expect(
            f"expr{jit_flag} -- float4{{1.25, 2.5, 3.25, 4.5}}",
            substrs=["(1.25, 2.5, 3.25, 4.5)"],
        )
        self.expect(
            f"expr{jit_flag} -- float4{{0.1, 0.2, 0.3, 0.4}}[0]", substrs=["0.1"]
        )

        oldValue = v.GetChildAtIndex(0).GetValue()
        v.SetFormat(lldb.eFormatHex)
        newValue = v.GetChildAtIndex(0).GetValue()
        self.assertNotEqual(
            oldValue, newValue, "values did not change along with format"
        )

        v.SetFormat(lldb.eFormatVectorOfFloat32)
        oldValueAgain = v.GetChildAtIndex(0).GetValue()
        self.assertEqual(oldValue, oldValueAgain, "same format but different values")

        # Test formatter for vector types whose size is not a power-of-2
        f3 = self.frame().FindVariable("f3")
        self.assertEqual(f3.GetNumChildren(), 3)
        self.assertEqual(f3.GetChildAtIndex(0).GetData().float[0], 1.25)
        self.assertEqual(f3.GetChildAtIndex(1).GetData().float[0], 2.50)
        self.assertEqual(f3.GetChildAtIndex(2).GetData().float[0], 2.50)

    # rdar://problem/14035604
    @skipIf(compiler="gcc")  # gcc don't have ext_vector_type extension
    def test_with_run_command(self):
        self.setup_and_run_to_breakpoint()
        self.runCmd("settings set plugin.jit-loader.gdb.enable on")
        self.vector_formatting_test()

    @skipIf(compiler="gcc")  # gcc don't have ext_vector_type extension
    def test_with_run_command_no_jit(self):
        self.setup_and_run_to_breakpoint()
        self.vector_formatting_test(allow_jit=False)
