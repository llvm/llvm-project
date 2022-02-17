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
        self.assertEqual(addr.function.name, name, "Got the right name")
        
    def test_break_on_overload(self):
        self.build()
        self.target = lldbutil.run_to_breakpoint_make_target(self)
        self.check_breakpoint("a_function(int)")
        self.check_breakpoint("a_function(double)")
        self.check_breakpoint("a_function(int, double)")
        self.check_breakpoint("a_function(double, int)")
        
        
        
