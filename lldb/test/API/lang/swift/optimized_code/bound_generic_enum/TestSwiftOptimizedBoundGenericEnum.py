import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftOptimizedBoundGenericEnum(lldbtest.TestBase):

    @swiftTest
    def test(self):
        """Test the bound generic enum types in "optimized" code."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(self,
            'break one', lldb.SBFileSpec('main.swift'))
        bkpt_two = target.BreakpointCreateBySourceRegex(
            'break two', lldb.SBFileSpec('main.swift'))
        self.assertGreater(bkpt_two.GetNumLocations(), 0)


        var_self = self.frame().FindVariable("self")
        # FIXME, this fails with a data extractor error.
        lldbutil.check_variable(self, var_self, False, value=None)
        lldbutil.continue_to_breakpoint(process, bkpt_two)
        var_self = self.frame().FindVariable("self")
        lldbutil.check_variable(self, var_self, True, value="success")
