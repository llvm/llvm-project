"""
When using C++11 in place member initialization, show that we
can set and hit breakpoints on initialization lines.  This is a
little bit tricky because we try not to move file and line breakpoints 
across function boundaries but these lines are outside the source range
of the constructor.
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_breakpoints_on_initializers(self):
        """Show we can set breakpoints on initializers appearing both before
           and after the constructor body, and hit them."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.cpp")
        self.first_initializer_line = line_number("main.cpp", "Set the before constructor breakpoint here")
        self.second_initializer_line = line_number("main.cpp", "Set the after constructor breakpoint here")

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   " Set a breakpoint here to get started", self.main_source_file)

        # Now set breakpoints on the two initializer lines we found in the test startup:
        bkpt1 = target.BreakpointCreateByLocation(self.main_source_file, self.first_initializer_line)
        self.assertEqual(bkpt1.GetNumLocations(), 1)
        bkpt2 = target.BreakpointCreateByLocation(self.main_source_file, self.second_initializer_line)
        self.assertEqual(bkpt2.GetNumLocations(), 1)

        # Now continue, we should stop at the two breakpoints above, first the one before, then
        # the one after.
        self.assertEqual(len(lldbutil.continue_to_breakpoint(process, bkpt1)), 1, "Hit first breakpoint")
        self.assertEqual(len(lldbutil.continue_to_breakpoint(process, bkpt2)), 1, "Hit second breakpoint")
        

