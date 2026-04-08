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
        # There's a bug in the Xcode 15.2 linker, where it did not emit
        # "function starts" entries for the branch island symbols, which
        # causes lldb to set the range of the previous non-island symbol to
        # encompass the range of the branch island symbols.  If we encounter
        # that bug, then we won't successfully do the step in.  Test for
        # that here - if the symbol doesn't round-trip from
        # name->address->name then the rest of the test can't pass.
        island_sym_ctx = syms[0]
        sym_addr = island_sym_ctx.symbol.addr
        resolved_name = sym_addr.symbol.name
        if resolved_name != "foo.island":
            self.skipTest("Encountered overlapping symbol linker bug")
        thread.StepInto()
        stop_frame = thread.frames[0]
        self.assertIn("foo", stop_frame.name, "Stepped into foo")
        var = stop_frame.FindVariable("a_variable_in_foo")
        self.assertTrue(var.IsValid(), "Found the variable in foo")
