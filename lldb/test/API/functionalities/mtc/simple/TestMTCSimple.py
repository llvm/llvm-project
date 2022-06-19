"""
Tests basic Main Thread Checker support (detecting a main-thread-only violation).
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbplatformutil import *
import json


class MTCSimpleTestCase(TestBase):

    @skipUnlessDarwin
    @skipIf(compiler="clang", compiler_version=['<', '9.0'])
    def test(self):
        self.mtc_dylib_path = findMainThreadCheckerDylib()
        if self.mtc_dylib_path == "":
            self.skipTest("This test requires libMainThreadChecker.dylib")

        self.build()
        self.mtc_tests()

    @skipIf(archs=['i386'])
    @skipIf(compiler="clang", compiler_version=['<', '9.0'])
    def mtc_tests(self):
        self.assertNotEqual(self.mtc_dylib_path, "")

        # Load the test
        exe = self.getBuildArtifact("a.out")
        self.expect("file " + exe, patterns=["Current executable set to .*a.out"])

        self.runCmd("env DYLD_INSERT_LIBRARIES=%s" % self.mtc_dylib_path)
        self.runCmd("run")

        process = self.dbg.GetSelectedTarget().process
        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()

        view = "NSView" if lldbplatformutil.getPlatform() == "macosx" else "UIView"

        self.expect("thread info",
                    substrs=['stop reason = -[' + view +
                             ' superview] must be used from main thread only'])

        self.expect(
            "thread info -s",
            substrs=[
                "api_name",
                "class_name",
                "description",
                "instrumentation_class",
                "selector"
            ])
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonInstrumentation)
        output_lines = self.res.GetOutput().split('\n')
        json_line = '\n'.join(output_lines[2:])
        data = json.loads(json_line)
        self.assertEqual(data["instrumentation_class"], "MainThreadChecker")
        self.assertEqual(data["api_name"], "-[" + view + " superview]")
        self.assertEqual(data["class_name"], view)
        self.assertEqual(data["selector"], "superview")
        self.assertEqual(data["description"], "-[" + view + " superview] must be used from main thread only")
