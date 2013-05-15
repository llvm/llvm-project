"""
Test more expression command sequences with objective-c.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class FoundationTestCase2(TestBase):

    mydir = os.path.join("lang", "objc", "foundation")

    @dsym_test
    def test_more_expr_commands_with_dsym(self):
        """More expression commands for objective-c."""
        self.buildDsym()
        self.more_expr_objc()

    @dwarf_test
    def test_more_expr_commands_with_dwarf(self):
        """More expression commands for objective-c."""
        self.buildDwarf()
        self.more_expr_objc()

    @dsym_test
    def test_NSArray_expr_commands_with_dsym(self):
        """Test expression commands for NSArray."""
        self.buildDsym()
        self.NSArray_expr()

    @dwarf_test
    def test_NSArray_expr_commands_with_dwarf(self):
        """Test expression commands for NSArray."""
        self.buildDwarf()
        self.NSArray_expr()

    @dsym_test
    def test_NSString_expr_commands_with_dsym(self):
        """Test expression commands for NSString."""
        self.buildDsym()
        self.NSString_expr()

    @dwarf_test
    def test_NSString_expr_commands_with_dwarf(self):
        """Test expression commands for NSString."""
        self.buildDwarf()
        self.NSString_expr()

    @dsym_test
    def test_MyString_dump_with_dsym(self):
        """Test dump of a known Objective-C object by dereferencing it."""
        self.buildDsym()
        self.MyString_dump()

    @dwarf_test
    def test_MyString_dump_with_dwarf(self):
        """Test dump of a known Objective-C object by dereferencing it."""
        self.buildDwarf()
        self.MyString_dump()

    @expectedFailurei386
    @dsym_test
    def test_NSError_po_with_dsym(self):
        """Test that po of the result of an unknown method doesn't require a cast."""
        self.buildDsym()
        self.NSError_po()

    @expectedFailurei386
    @dwarf_test
    def test_NSError_po_with_dwarf(self):
        """Test that po of the result of an unknown method doesn't require a cast."""
        self.buildDsym()
        self.NSError_po()
        
    @dsym_test
    def test_NSError_p_with_dsym(self):
        """Test that p of the result of an unknown method does require a cast."""
        self.buildDsym()
        self.NSError_p()

    @dwarf_test
    def test_NSError_p_with_dwarf(self):
        """Test that p of the result of an unknown method does require a cast."""
        self.buildDsym()
        self.NSError_p()
                
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break at.
        self.lines = []
        self.lines.append(line_number('main.m', '// Break here for selector: tests'))
        self.lines.append(line_number('main.m', '// Break here for NSArray tests'))
        self.lines.append(line_number('main.m', '// Break here for NSString tests'))
        self.lines.append(line_number('main.m', '// Break here for description test'))
        self.lines.append(line_number('main.m', '// Set break point at this line'))
    
    def more_expr_objc(self):
        """More expression commands for objective-c."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Create a bunch of breakpoints.
        for line in self.lines:
            lldbutil.run_break_set_by_file_and_line (self, "main.m", line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Test_Selector:
        self.runCmd("thread backtrace")
        self.expect("expression (char *)sel_getName(sel)",
            substrs = ["(char *)",
                       "length"])

        self.runCmd("process continue")

        # Test_NSArray:
        self.runCmd("thread backtrace")
        self.runCmd("process continue")

        # Test_NSString:
        self.runCmd("thread backtrace")
        self.runCmd("process continue")

        # Test_MyString:
        self.runCmd("thread backtrace")
        self.expect("expression (char *)sel_getName(_cmd)",
            substrs = ["(char *)",
                       "description"])

        self.runCmd("process continue")

    @unittest2.expectedFailure(8741897)
    def NSArray_expr(self):
        """Test expression commands for NSArray."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside Test_NSArray:
        line = self.lines[1]
        lldbutil.run_break_set_by_file_and_line (self, "main.m", line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Test_NSArray:
        self.runCmd("thread backtrace")
        self.expect("expression (int)[nil_mutable_array count]",
            patterns = ["\(int\) \$.* = 0"])
        self.expect("expression (int)[array1 count]",
            patterns = ["\(int\) \$.* = 3"])
        self.expect("expression (int)[array2 count]",
            patterns = ["\(int\) \$.* = 3"])
        self.expect("expression (int)array1.count",
            patterns = ["\(int\) \$.* = 3"])
        self.expect("expression (int)array2.count",
            patterns = ["\(int\) \$.* = 3"])
        self.runCmd("process continue")

    @unittest2.expectedFailure(8741897)
    def NSString_expr(self):
        """Test expression commands for NSString."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside Test_NSString:
        line = self.lines[2]
        lldbutil.run_break_set_by_file_and_line (self, "main.m", line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Test_NSString:
        self.runCmd("thread backtrace")
        self.expect("expression (int)[str length]",
            patterns = ["\(int\) \$.* ="])
        self.expect("expression (int)[str_id length]",
            patterns = ["\(int\) \$.* ="])
        self.expect("expression [str description]",
            patterns = ["\(id\) \$.* = 0x"])
        self.expect("expression [str_id description]",
            patterns = ["\(id\) \$.* = 0x"])
        self.expect("expression str.description")
        self.expect("expression str_id.description")
        self.expect('expression str = @"new"')
        self.expect('expression str = [NSString stringWithFormat: @"%cew", \'N\']')
        self.runCmd("process continue")

    def MyString_dump(self):
        """Test dump of a known Objective-C object by dereferencing it."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        
        line = self.lines[4]

        lldbutil.run_break_set_by_file_and_line (self, "main.m", line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        
        self.expect("expression --show-types -- *my",
            patterns = ["\(MyString\) \$.* = ", "\(MyBase\)", "\(NSObject\)", "\(Class\)"])
        self.runCmd("process continue")

    def NSError_po(self):
        """Test that po of the result of an unknown method doesn't require a cast."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        
        line = self.lines[4]

        lldbutil.run_break_set_by_file_and_line (self, "main.m", line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("po [NSError errorWithDomain:@\"Hello\" code:35 userInfo:nil]",
            substrs = ["Error Domain=Hello", "Code=35", "be completed."])
        self.runCmd("process continue")

    def NSError_p(self):
        """Test that p of the result of an unknown method does require a cast."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        
        line = self.lines[4]

        lldbutil.run_break_set_by_file_and_line (self, "main.m", line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("p [NSError thisMethodIsntImplemented:0]",
                    error = True, 
                    patterns = ["no known method", "cast the message send to the method's return type"])
        self.runCmd("process continue")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
