"""
Regression test for <rdar://problem/8981098>:

The expression parser's type search only looks in the current compilation unit for types.
"""

import unittest2
import lldb
from lldbtest import *
import lldbutil

class ObjCTypeQueryTestCase(TestBase):

    mydir = os.path.join("expression_command", "two-files")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.m.
        self.line = line_number('main.m',
                                "// Set breakpoint here, then do 'expr (NSArray*)array_token'.")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """The expression parser's type search should be wider than the current compilation unit."""
        self.buildDsym()
        self.type_query_from_other_cu()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_with_dwarf(self):
        """The expression parser's type search should be wider than the current compilation unit."""
        self.buildDwarf()
        self.type_query_from_other_cu()

    def type_query_from_other_cu(self):
        """The expression parser's type search should be wider than the current compilation unit."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Now do a NSArry type query from the 'main.m' compile uint.
        self.expect("expression (NSArray*)array_token",
            substrs = ['(NSArray *) $0 = 0x'])
        # (NSArray *) $0 = 0x00007fff70118398


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
