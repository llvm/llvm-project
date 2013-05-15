"""
Tests that rvalue references are supported in C++
"""

import lldb
from lldbtest import *
import lldbutil

class CPPThisTestCase(TestBase):
    
    mydir = os.path.join("lang", "cpp", "rvalue-references")
    
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    #rdar://problem/11479676
    @expectedFailureClang
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test that rvalues are supported in the C++ expression parser"""
        self.buildDsym()
        self.static_method_commands()

    #rdar://problem/11479676
    @expectedFailureClang
    @expectedFailureGcc # GCC (4.7) does not emit correct DWARF tags for rvalue-references
    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Test that rvalues are supported in the C++ expression parser"""
        self.buildDwarf()
        self.static_method_commands()

    def setUp(self):
        TestBase.setUp(self)
    
    def set_breakpoint(self, line):
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", line, num_expected_locations=1, loc_exact=True)

    def static_method_commands(self):
        """Test that rvalues are supported in the C++ expression parser"""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.set_breakpoint(line_number('main.cpp', '// breakpoint 1'))
        self.set_breakpoint(line_number('main.cpp', '// breakpoint 2'))

        self.runCmd("process launch", RUN_SUCCEEDED)

        self.expect("expression -- i",
                    startstr = "(int &&) $0 =",
                    substrs = ["3"])

        self.expect("breakpoint delete 1")

        self.runCmd("process continue")
        
        self.expect("expression -- foo(2)")

        self.expect("expression -- int &&j = 3; foo(j)",
                    error = True)

        self.expect("expression -- int &&k = 6; k",
                    startstr = "(int) $1 = 6")
        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
