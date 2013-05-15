"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class SmartArrayDataFormatterTestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "data-formatter-smart-array")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test data formatter commands."""
        self.buildDsym()
        self.data_formatter_commands()

    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Test data formatter commands."""
        self.buildDwarf()
        self.data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def data_formatter_commands(self):
        """Test that that file and class static variables display correctly."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])
        
        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

# check that we are not looping here
        self.runCmd("type summary add --summary-string \"${var%V}\" SomeData")

        self.expect("frame variable data",
            substrs = ['invalid use of aggregate type'])
# ${var%s}
        self.runCmd("type summary add --summary-string \"ptr = ${var%s}\" \"char *\"")

        self.expect("frame variable strptr",
            substrs = ['ptr = \"',
                       'Hello world!'])

        self.expect("frame variable other.strptr",
            substrs = ['ptr = \"',
                        'Nested Hello world!'])
        
        self.runCmd("type summary add --summary-string \"arr = ${var%s}\" -x \"char \\[[0-9]+\\]\"")
        
        self.expect("frame variable strarr",
                    substrs = ['arr = \"',
                               'Hello world!'])
        
        self.expect("frame variable other.strarr",
                    substrs = ['arr = \"',
                               'Nested Hello world!'])

        self.expect("p strarr",
                    substrs = ['arr = \"',
                               'Hello world!'])

        self.expect("p other.strarr",
                    substrs = ['arr = \"',
                               'Nested Hello world!'])

# ${var%c}
        self.runCmd("type summary add --summary-string \"ptr = ${var%c}\" \"char *\"")
    
        self.expect("frame variable strptr",
                substrs = ['ptr = \"',
                           'Hello world!'])
    
        self.expect("frame variable other.strptr",
                substrs = ['ptr = \"',
                           'Nested Hello world!'])

        self.expect("p strptr",
                    substrs = ['ptr = \"',
                               'Hello world!'])

        self.expect("p other.strptr",
                    substrs = ['ptr = \"',
                               'Nested Hello world!'])

        self.runCmd("type summary add --summary-string \"arr = ${var%c}\" -x \"char \\[[0-9]+\\]\"")

        self.expect("frame variable strarr",
                    substrs = ['arr = \"',
                               'Hello world!'])

        self.expect("frame variable other.strarr",
                    substrs = ['arr = \"',
                               'Nested Hello world!'])
        
        self.expect("p strarr",
                    substrs = ['arr = \"',
                               'Hello world!'])

        self.expect("p other.strarr",
                    substrs = ['arr = \"',
                               'Nested Hello world!'])

# ${var%char[]}
        self.runCmd("type summary add --summary-string \"arr = ${var%char[]}\" -x \"char \\[[0-9]+\\]\"")

        self.expect("frame variable strarr",
                    substrs = ['arr = \"',
                               'Hello world!'])

        self.expect("frame variable other.strarr",
                    substrs = ['arr = ',
                               'Nested Hello world!'])

        self.expect("p strarr",
                    substrs = ['arr = \"',
                               'Hello world!'])

        self.expect("p other.strarr",
                    substrs = ['arr = ',
                               'Nested Hello world!'])

        self.runCmd("type summary add --summary-string \"ptr = ${var%char[]}\" \"char *\"")

        self.expect("frame variable strptr",
            substrs = ['ptr = \"',
            'Hello world!'])
        
        self.expect("frame variable other.strptr",
            substrs = ['ptr = \"',
            'Nested Hello world!'])

        self.expect("p strptr",
                    substrs = ['ptr = \"',
                               'Hello world!'])

        self.expect("p other.strptr",
                    substrs = ['ptr = \"',
                               'Nested Hello world!'])

# ${var%a}
        self.runCmd("type summary add --summary-string \"arr = ${var%a}\" -x \"char \\[[0-9]+\\]\"")

        self.expect("frame variable strarr",
                    substrs = ['arr = \"',
                               'Hello world!'])

        self.expect("frame variable other.strarr",
                    substrs = ['arr = ',
                               'Nested Hello world!'])

        self.expect("p strarr",
                    substrs = ['arr = \"',
                               'Hello world!'])

        self.expect("p other.strarr",
                    substrs = ['arr = ',
                               'Nested Hello world!'])

        self.runCmd("type summary add --summary-string \"ptr = ${var%a}\" \"char *\"")

        self.expect("frame variable strptr",
                    substrs = ['ptr = \"',
                               'Hello world!'])

        self.expect("frame variable other.strptr",
                    substrs = ['ptr = \"',
                               'Nested Hello world!'])

        self.expect("p strptr",
                    substrs = ['ptr = \"',
                               'Hello world!'])

        self.expect("p other.strptr",
                    substrs = ['ptr = \"',
                               'Nested Hello world!'])

        self.runCmd("type summary add --summary-string \"ptr = ${var[]%char[]}\" \"char *\"")
        
# I do not know the size of the data, but you are asking for a full array slice..
# use the ${var%char[]} to obtain a string as result
        self.expect("frame variable strptr", matching=False,
                    substrs = ['ptr = \"',
                               'Hello world!'])
        
        self.expect("frame variable other.strptr", matching=False,
                    substrs = ['ptr = \"',
                               'Nested Hello world!'])

        self.expect("p strptr", matching=False,
                    substrs = ['ptr = \"',
                               'Hello world!'])

        self.expect("p other.strptr", matching=False,
                    substrs = ['ptr = \"',
                               'Nested Hello world!'])

# You asked an array-style printout...
        self.runCmd("type summary add --summary-string \"ptr = ${var[0-1]%char[]}\" \"char *\"")
        
        self.expect("frame variable strptr",
                    substrs = ['ptr = ',
                               '[{H},{e}]'])
        
        self.expect("frame variable other.strptr",
                    substrs = ['ptr = ',
                               '[{N},{e}]'])

        self.expect("p strptr",
                    substrs = ['ptr = ',
                               '[{H},{e}]'])

        self.expect("p other.strptr",
                    substrs = ['ptr = ',
                               '[{N},{e}]'])

# using [] is required here
        self.runCmd("type summary add --summary-string \"arr = ${var%x}\" \"int [5]\"")
        
        self.expect("frame variable intarr",matching=False,
                    substrs = ['0x00000001,0x00000001,0x00000002,0x00000003,0x00000005'])
        
        self.expect("frame variable other.intarr", matching=False,
                    substrs = ['0x00000009,0x00000008,0x00000007,0x00000006,0x00000005'])

        self.runCmd("type summary add --summary-string \"arr = ${var[]%x}\" \"int [5]\"")
        
        self.expect("frame variable intarr",
                    substrs = ['intarr = arr =',
                               '0x00000001,0x00000001,0x00000002,0x00000003,0x00000005'])
        
        self.expect("frame variable other.intarr",
                    substrs = ['intarr = arr =',
                               '0x00000009,0x00000008,0x00000007,0x00000006,0x00000005'])

# printing each array item as an array
        self.runCmd("type summary add --summary-string \"arr = ${var[]%uint32_t[]}\" \"int [5]\"")
        
        self.expect("frame variable intarr",
                    substrs = ['intarr = arr =',
                               '{0x00000001},{0x00000001},{0x00000002},{0x00000003},{0x00000005}'])
        
        self.expect("frame variable other.intarr",
                    substrs = ['intarr = arr = ',
                               '{0x00000009},{0x00000008},{0x00000007},{0x00000006},{0x00000005}'])

# printing full array as an array
        self.runCmd("type summary add --summary-string \"arr = ${var%uint32_t[]}\" \"int [5]\"")
        
        self.expect("frame variable intarr",
                    substrs = ['intarr = arr =',
                               '0x00000001,0x00000001,0x00000002,0x00000003,0x00000005'])

        self.expect("frame variable other.intarr",
                    substrs = ['intarr = arr =',
                               '0x00000009,0x00000008,0x00000007,0x00000006,0x00000005'])

# printing each array item as an array
        self.runCmd("type summary add --summary-string \"arr = ${var[]%float32[]}\" \"float [7]\"")
        
        self.expect("frame variable flarr",
                    substrs = ['flarr = arr =',
                               '{78.5},{77.4},{78},{76.1},{76.7},{76.8},{77}'])
        
        self.expect("frame variable other.flarr",
                    substrs = ['flarr = arr = ',
                               '{25.5},{25.7},{25.9},{26.4},{27.1},{27.3},{26.9}'])
        
# printing full array as an array
        self.runCmd("type summary add --summary-string \"arr = ${var%float32[]}\" \"float [7]\"")
        
        self.expect("frame variable flarr",
                    substrs = ['flarr = arr =',
                               '78.5,77.4,78,76.1,76.7,76.8,77'])
        
        self.expect("frame variable other.flarr",
                    substrs = ['flarr = arr =',
                               '25.5,25.7,25.9,26.4,27.1,27.3,26.9'])

# using array smart summary strings for pointers should make no sense
        self.runCmd("type summary add --summary-string \"arr = ${var%float32[]}\" \"float *\"")
        self.runCmd("type summary add --summary-string \"arr = ${var%int32_t[]}\" \"int *\"")

        self.expect("frame variable flptr", matching=False,
                    substrs = ['78.5,77.4,78,76.1,76.7,76.8,77'])
        
        self.expect("frame variable intptr", matching=False,
                    substrs = ['1,1,2,3,5'])

# use y and Y
        self.runCmd("type summary add --summary-string \"arr = ${var%y}\" \"float [7]\"")
        self.runCmd("type summary add --summary-string \"arr = ${var%y}\" \"int [5]\"")

        self.expect("frame variable flarr",
                    substrs = ['flarr = arr =',
                               '00 00 9d 42,cd cc 9a 42,00 00 9c 42,33 33 98 42,66 66 99 42,9a 99 99 42,00 00 9a 42'])
        
        self.expect("frame variable other.flarr",
                    substrs = ['flarr = arr =',
                               '00 00 cc 41,9a 99 cd 41,33 33 cf 41,33 33 d3 41,cd cc d8 41,66 66 da 41,33 33 d7 41'])

        self.expect("frame variable intarr",
                    substrs = ['intarr = arr =',
                               '01 00 00 00,01 00 00 00,02 00 00 00,03 00 00 00,05 00 00 00'])
        
        self.expect("frame variable other.intarr",
                    substrs = ['intarr = arr = ',
                               '09 00 00 00,08 00 00 00,07 00 00 00,06 00 00 00,05 00 00 00'])
                    
        self.runCmd("type summary add --summary-string \"arr = ${var%Y}\" \"float [7]\"")
        self.runCmd("type summary add --summary-string \"arr = ${var%Y}\" \"int [5]\"")
            
        self.expect("frame variable flarr",
                    substrs = ['flarr = arr =',
                               '...B,cd cc 9a 42',
                               'ff.B,9a 99 99 42'])
        
        self.expect("frame variable other.flarr",
                    substrs = ['flarr = arr =',
                               '...A,33 33 cf 41',
                               '33.A,cd cc d8 41'])
        
        self.expect("frame variable intarr",
                    substrs = ['intarr = arr =',
                               '....,01 00 00 00',
                               '....,05 00 00 00'])
        
        self.expect("frame variable other.intarr",
                    substrs = ['intarr = arr = ',
                               '09 00 00 00',
                               '....,07 00 00 00'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
