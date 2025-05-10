"""
Make sure that we can step in across an arm64 branch island
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestBranchIslandStepping(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessAppleSilicon
    def test_step_in_branch_island(self):
        """Make sure we can step in across a branch island"""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.do_test()

    def do_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", self.main_source_file
        )

        # Make sure that we did manage to generate a branch island for foo:
        syms = target.FindSymbols("foo.island", lldb.eSymbolTypeCode)
        self.assertEqual(len(syms), 1, "We did generate an island for foo")

        thread.StepInto()
        stop_frame = thread.frames[0]
        self.assertIn("foo", stop_frame.name, "Stepped into foo")
        var = stop_frame.FindVariable("a_variable_in_foo")
        self.assertTrue(var.IsValid(), "Found the variable in foo")
