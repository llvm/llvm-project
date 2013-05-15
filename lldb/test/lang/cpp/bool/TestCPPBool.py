"""
Tests that bool types work
"""
import lldb
from lldbtest import *
import lldbutil

class CPPBoolTestCase(TestBase):
    
    mydir = os.path.join("lang", "cpp", "bool")
    
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test that bool types work in the expression parser"""
        self.buildDsym()
        self.static_method_commands()

    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Test that bool types work in the expression parser"""
        self.buildDwarf()
        self.static_method_commands()

    def setUp(self):
        TestBase.setUp(self)
    
    def set_breakpoint(self, line):
        # Some compilers (for example GCC 4.4.7 and 4.6.1) emit multiple locations for the statement with the ternary
        # operator in the test program, while others emit only 1.
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", line, num_expected_locations=-1, loc_exact=False)

    def static_method_commands(self):
        """Test that bool types work in the expression parser"""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.set_breakpoint(line_number('main.cpp', '// breakpoint 1'))

        self.runCmd("process launch", RUN_SUCCEEDED)

        self.expect("expression -- bool second_bool = my_bool; second_bool",
                    startstr = "(bool) $0 = false")

        self.expect("expression -- my_bool = true",
                    startstr = "(bool) $1 = true")
        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
