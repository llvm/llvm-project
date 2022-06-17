"""
Test that hitting a UBSan issue while running user expression doesn't break the evaluation.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class UbsanUserExpressionTestCase(TestBase):

    @skipUnlessUndefinedBehaviorSanitizer
    def test(self):
        self.build()
        self.ubsan_tests()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.line_breakpoint = line_number('main.c', '// breakpoint line')

    def ubsan_tests(self):
        # Load the test
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.registerSanitizerLibrariesWithTarget(target)

        self.runCmd("breakpoint set -f main.c -l %d" % self.line_breakpoint)

        self.runCmd("run")

        process = self.dbg.GetSelectedTarget().process
        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()

        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        self.expect("p foo()", substrs=["(int) $0 = 42"])

        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])
