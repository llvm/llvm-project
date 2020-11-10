"""
Test more expression command sequences with objective-c.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class FoundationTestCase2(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def test_expr_commands(self):
        """More expression commands for objective-c."""
        self.build()
        main_spec = lldb.SBFileSpec("main.m")

        (target, process, thread, bp) = lldbutil.run_to_source_breakpoint(
            self, "Break here for selector: tests", main_spec)
        
        # Test_Selector:
        self.expect("expression (char *)sel_getName(sel)",
                    substrs=["(char *)",
                             "length"])

        desc_bkpt = target.BreakpointCreateBySourceRegex("Break here for description test",
                                                          main_spec)
        self.assertEqual(desc_bkpt.GetNumLocations(), 1, "description breakpoint has a location")
        lldbutil.continue_to_breakpoint(process, desc_bkpt)
        
        self.expect("expression (char *)sel_getName(_cmd)",
                    substrs=["(char *)",
                             "description"])
