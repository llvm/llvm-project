"""
Tests Main Thread Checker support on Swift code.
"""

import os
import time
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbplatformutil import *
import json


class MTCSwiftTestCase(TestBase):
    @expectedFailureAll(bugnumber="rdar://60396797",
                        setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.mtc_dylib_path = findMainThreadCheckerDylib()
        self.assertTrue(self.mtc_dylib_path != "")
        self.build()
        self.mtc_tests()

    def mtc_tests(self):
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
                    substrs=['stop reason = ' + view +
                             '.removeFromSuperview() must be used from main thread only'])

        self.expect(
            "thread info -s",
            ordered=False,
            substrs=["instrumentation_class", "api_name", "class_name", "selector", "description"])
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonInstrumentation)
        output_lines = self.res.GetOutput().split('\n')
        json_line = '\n'.join(output_lines[2:])
        data = json.loads(json_line)
        self.assertEqual(data["instrumentation_class"], "MainThreadChecker")
        self.assertEqual(data["api_name"], view + ".removeFromSuperview()")
        self.assertEqual(data["class_name"], view)
        self.assertEqual(data["selector"], "removeFromSuperview")
        self.assertEqual(data["description"], view + ".removeFromSuperview() must be used from main thread only")
