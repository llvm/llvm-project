"""
Test thread step-in [ -r | --step-over-regexp ].
"""



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *


class ThreadStepInAvoidRegexTestCase(TestBase):

    def setUp(self):
        TestBase.setUp(self)
        self.line2 = line_number('main.c', '// assignment to B2')

    @skipIfWindows
    def test_step_out_avoid_regexp(self):
        """Exercise thread step-in -r"""
        self.build()
        lldbutil.run_to_source_breakpoint(self,
                'frame select 2, thread step-out while stopped',
                lldb.SBFileSpec('main.c'))

        # Now step in, skipping the frames for 'b' and 'a'.
        self.runCmd("thread step-in -r 'a'")

        # We should be at the assignment to B2.
        self.expect("thread backtrace", STEP_IN_SUCCEEDED,
                    substrs=["stop reason = step in"],
                    patterns=["frame #0.*main.c:%d" % self.line2])
