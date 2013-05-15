"""
Tests that ObjC member variables are available where they should be.
"""
import lldb
from lldbtest import *
import lldbutil

class ObjCSelfTestCase(TestBase):
    
    mydir = os.path.join("lang", "objc", "self")
    
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test that the appropriate member variables are available when stopped in Objective-C class and instance methods"""
        self.buildDsym()
        self.self_commands()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Test that the appropriate member variables are available when stopped in Objective-C class and instance methods"""
        self.buildDwarf()
        self.self_commands()

    def setUp(self):
        TestBase.setUp(self)
    
    def set_breakpoint(self, line):
        lldbutil.run_break_set_by_file_and_line (self, "main.m", line, num_expected_locations=1, loc_exact=True)
    
    def self_commands(self):
        """Test that the appropriate member variables are available when stopped in Objective-C class and instance methods"""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.set_breakpoint(line_number('main.m', '// breakpoint 1'))
        self.set_breakpoint(line_number('main.m', '// breakpoint 2'))

        self.runCmd("process launch", RUN_SUCCEEDED)

        self.expect("expression -- m_a = 2",
                    startstr = "(int) $0 = 2")
        
        self.runCmd("process continue")
        
        # This would be disallowed if we enforced const.  But we don't.
        self.expect("expression -- m_a = 2",
                    error=True)
        
        self.expect("expression -- s_a", 
                    startstr = "(int) $1 = 5")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
