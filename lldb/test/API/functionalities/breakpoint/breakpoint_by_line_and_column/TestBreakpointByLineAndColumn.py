"""
Test setting a breakpoint by line and column.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BreakpointByLineAndColumnTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    ## Skip gcc version less 7.1 since it doesn't support -gcolumn-info
    @skipIf(compiler="gcc", compiler_version=['<', '7.1'])
    def testBreakpointByLineAndColumn(self):
        self.build()
        main_c = lldb.SBFileSpec("main.c")
        _, _, _, breakpoint = lldbutil.run_to_line_breakpoint(self,
                                                              main_c, 11, 50)
        self.expect("fr v did_call", substrs=['1'])
        in_then = False
        for i in range(breakpoint.GetNumLocations()):
            b_loc = breakpoint.GetLocationAtIndex(i).GetAddress().GetLineEntry()
            self.assertEqual(b_loc.GetLine(), 11)
            in_then |= b_loc.GetColumn() == 50
        self.assertTrue(in_then)

    ## Skip gcc version less 7.1 since it doesn't support -gcolumn-info
    @skipIf(compiler="gcc", compiler_version=['<', '7.1'])
    def testBreakpointByLine(self):
        self.build()
        main_c = lldb.SBFileSpec("main.c")
        _, _, _, breakpoint = lldbutil.run_to_line_breakpoint(self, main_c, 11)
        self.expect("fr v did_call", substrs=['0'])
        in_condition = False
        for i in range(breakpoint.GetNumLocations()):
            b_loc = breakpoint.GetLocationAtIndex(i).GetAddress().GetLineEntry()
            self.assertEqual(b_loc.GetLine(), 11)
            in_condition |= b_loc.GetColumn() < 30
        self.assertTrue(in_condition)

    ## Skip gcc version less 7.1 since it doesn't support -gcolumn-info
    @skipIf(compiler="gcc", compiler_version=['<', '7.1'])
    def testBreakpointByLineAndColumnNearestCode(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        main_c = lldb.SBFileSpec("main.c")
        line = line_number("main.c", "// Line 20.")
        column = len("// Line 20") # should stop at the period.
        indent = 2
        module_list = lldb.SBFileSpecList()

        # Create a target from the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        valid_bpt = target.BreakpointCreateByLocation(main_c, line, column,
                                                      indent, module_list, True)
        self.assertTrue(valid_bpt, VALID_BREAKPOINT)
        self.assertEqual(valid_bpt.GetNumLocations(), 1)

        invalid_bpt = target.BreakpointCreateByLocation(main_c, line, column,
                                                      indent, module_list, False)
        self.assertTrue(invalid_bpt, VALID_BREAKPOINT)
        self.assertEqual(invalid_bpt.GetNumLocations(), 0)

