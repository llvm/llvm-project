"""
Test thread until in a recursive function.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestStepUntilRecursive(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_until_after_recursive_call(self):
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// base case", lldb.SBFileSpec("main.c")
        )
        after_call = line_number("main.c", "// after recursive call")
        self.assertSuccess(
            thread.StepOverUntil(
                thread.GetFrameAtIndex(1), lldb.SBFileSpec("main.c"), after_call
            )
        )
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.GetLineEntry().GetLine(), after_call)
        self.assertEqual(frame.FindVariable("n").GetValueAsSigned(), 1)
