"""
Test setting a breakpoint on an overloaded function by name.
"""

import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestBreakpointOnOverload(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def check_breakpoint(self, name):
        bkpt = self.target.BreakpointCreateByName(name)
        self.assertEqual(bkpt.num_locations, 1, "Got one location")
        addr = bkpt.locations[0].GetAddress()
        self.assertTrue(addr.function.IsValid(), "Got a real function")
        # On Window, the name of the function includes the return value.
        # We still succeed in setting the breakpoint, but the resultant
        # name is not the same.
        # So just look for the name we used for the breakpoint in the
        # function name, rather than doing an equality check.
        self.assertIn(name, addr.function.name, "Got the right name")
        
    def test_break_on_overload(self):
        self.build()
        self.target = lldbutil.run_to_breakpoint_make_target(self)
        self.check_breakpoint("a_function(int)")
        self.check_breakpoint("a_function(double)")
        self.check_breakpoint("a_function(int, double)")
        self.check_breakpoint("a_function(double, int)")
        
        
        
