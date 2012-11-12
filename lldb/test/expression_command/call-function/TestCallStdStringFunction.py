"""
Test calling std::String member functions.
"""

import unittest2
import lldb
import lldbutil
from lldbtest import *

class ExprCommandCallFunctionTestCase(TestBase):

    mydir = os.path.join("expression_command", "call-function")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.cpp',
                                '// Please test these expressions while stopped at this line:')

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test calling std::String member function."""
        self.buildDsym()
        self.call_function()

    @dwarf_test
    def test_with_dwarf(self):
        """Test calling std::String member function."""
        self.buildDsym()
        self.call_function()

    def call_function(self):
        """Test calling std::String member function."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("print str",
            substrs = ['Hello world'])

        # Should be fixed with r142717.
        #
        # rdar://problem/9471744 test failure: ./dotest.py -C clang -v -w -t -p CallStdString
        # runCmd: print str.c_str()
        # runCmd failed!
        # error: Couldn't convert the expression to DWARF
        self.expect("print str.c_str()",
            substrs = ['Hello world'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
