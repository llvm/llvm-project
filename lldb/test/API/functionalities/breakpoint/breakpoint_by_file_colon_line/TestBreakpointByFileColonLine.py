"""
Test setting a breakpoint by line and column.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BreakpointByLineAndColumnTestCase(TestBase):

    def testBreakpointSpecWithLine(self):
        self.build()
        target = self.createTestTarget()
        
        # This one should work:
        lldbutil.run_break_set_by_file_colon_line(self, "main.c:11", "main.c", 11, num_expected_locations = 1)
        # Let's try an illegal specifier to make sure the command fails.  I'm not being exhaustive
        # since the UnitTest has more bad patterns.  I'm just testing that if the SetFromString
        # fails, we propagate the error.
        self.expect("break set -y 'foo.c'", error=True)
        
    ## Skip gcc version less 7.1 since it doesn't support -gcolumn-info
    @skipIf(compiler="gcc", compiler_version=['<', '7.1'])
    def testBreakpointByLine(self):
        self.build()
        target = self.createTestTarget()

        main_c = lldb.SBFileSpec("main.c")
        lldbutil.run_break_set_by_file_colon_line(self, "main.c:11:50", "main.c", 11, num_expected_locations = 1)
        
