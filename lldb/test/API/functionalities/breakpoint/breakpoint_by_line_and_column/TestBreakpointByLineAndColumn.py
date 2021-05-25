"""
Test setting a breakpoint by line and column.
"""

import re
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
        src_file = lldb.SBFileSpec("main.cpp")
        line = line_number("main.cpp",
                           "At the beginning of a function name (col:50)") + 1 # Next line after comment
        _, _, _, breakpoint = lldbutil.run_to_line_breakpoint(self,
                                                              src_file, line, 50)
        self.expect("fr v did_call", substrs=['1'])
        in_then = False
        for i in range(breakpoint.GetNumLocations()):
            b_loc = breakpoint.GetLocationAtIndex(i).GetAddress().GetLineEntry()
            self.assertEqual(b_loc.GetLine(), line)
            in_then |= b_loc.GetColumn() == 50
        self.assertTrue(in_then)

    ## Skip gcc version less 7.1 since it doesn't support -gcolumn-info
    @skipIf(compiler="gcc", compiler_version=['<', '7.1'])
    def testBreakpointByLine(self):
        self.build()
        src_file = lldb.SBFileSpec("main.cpp")
        line = line_number("main.cpp",
                           "At the beginning of a function name (col:50)") + 1 # Next line after comment
        _, _, _, breakpoint = lldbutil.run_to_line_breakpoint(self, src_file,
                                                              line)
        self.expect("fr v did_call", substrs=['0'])
        in_condition = False
        for i in range(breakpoint.GetNumLocations()):
            b_loc = breakpoint.GetLocationAtIndex(i).GetAddress().GetLineEntry()
            self.assertEqual(b_loc.GetLine(), line)
            in_condition |= b_loc.GetColumn() < 30
        self.assertTrue(in_condition)

    @skipIfWindows
    ## Skip gcc version less 7.1 since it doesn't support -gcolumn-info
    @skipIf(compiler="gcc", compiler_version=['<', '7.1'])
    def testBreakpointByLineAndColumnNearestCode(self):
        self.build()

        patterns = [
            "In the middle of a function name (col:42)",
            "In the middle of the lambda declaration argument (col:23)",
            "Inside the lambda (col:26)"
        ]

        source_loc = []

        for pattern in patterns:
            line = line_number("main.cpp", pattern) + 1
            column = int(re.search('\(col:([0-9]+)\)', pattern).group(1))
            source_loc.append({'line':line, 'column':column})

        target = self.createTestTarget()

        for loc in source_loc:
            src_file = lldb.SBFileSpec("main.cpp")
            line = loc['line']
            column = loc['column']
            indent = 0
            module_list = lldb.SBFileSpecList()

            valid_bpkt = target.BreakpointCreateByLocation(src_file, line,
                                                          column, indent,
                                                          module_list, True)
            self.assertTrue(valid_bpkt, VALID_BREAKPOINT)
            self.assertEqual(valid_bpkt.GetNumLocations(), 1)

        process = target.LaunchSimple(
                            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        nearest_column = [7, 17, 26]

        for idx,loc in enumerate(source_loc):
            bpkt = target.GetBreakpointAtIndex(idx)
            bpkt_loc = bpkt.GetLocationAtIndex(0)
            self.assertEqual(bpkt_loc.GetHitCount(), 1)
            self.assertSuccess(process.Continue())
            bpkt_loc_desc = lldb.SBStream()
            self.assertTrue(bpkt_loc.GetDescription(bpkt_loc_desc, lldb.eDescriptionLevelVerbose))
            self.assertIn("main.cpp:{}:{}".format(loc['line'], nearest_column[idx]),
                          bpkt_loc_desc.GetData())
            bpkt_loc_addr = bpkt_loc.GetAddress()
            self.assertTrue(bpkt_loc_addr)

            list = target.FindCompileUnits(lldb.SBFileSpec("main.cpp", False))
            # Executable has been built just from one source file 'main.cpp',
            # so we may check only the first element of list.
            compile_unit = list[0].GetCompileUnit()

            found = False
            for line_entry in compile_unit:
                if line_entry.GetStartAddress() == bpkt_loc_addr:
                    self.assertEqual(line_entry.GetFileSpec().GetFilename(),
                                    "main.cpp")
                    self.assertEqual(line_entry.GetLine(), loc['line'])
                    self.assertEqual(line_entry.GetColumn(), nearest_column[idx])
                    found = True
                    break

            self.assertTrue(found)

        line = line_number("main.cpp", "// This is a random comment.")
        column = len("// This is a random comment.")
        indent = 2
        invalid_bpkt = target.BreakpointCreateByLocation(src_file, line, column,
                                                      indent, module_list, False)
        self.assertTrue(invalid_bpkt, VALID_BREAKPOINT)
        self.assertEqual(invalid_bpkt.GetNumLocations(), 0)

