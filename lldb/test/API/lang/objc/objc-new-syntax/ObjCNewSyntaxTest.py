"""Test that the Objective-C syntax for dictionary/array literals and indexing works"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCNewSyntaxTest(TestBase):

    def target(self):
        return self._target

    def runToBreakpoint(self):
        self.build()
        self._target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, '// Set breakpoint 0 here.', lldb.SBFileSpec(
                'main.m', False))

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=['stopped', 'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno = 1, expected_hit_count = 1)
