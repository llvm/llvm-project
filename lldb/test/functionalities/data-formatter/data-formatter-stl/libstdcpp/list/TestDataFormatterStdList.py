"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class StdListDataFormatterTestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "data-formatter-stl", "libstdcpp", "list")

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

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=-1)

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
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)
            self.runCmd("settings set target.max-children-count 256", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("frame variable numbers_list -T")
        #self.runCmd("type synth add std::int_list std::string_list int_list string_list -l StdListSynthProvider")
        self.runCmd("type summary add std::int_list std::string_list int_list string_list --summary-string \"list has ${svar%#} items\" -e")
        self.runCmd("type format add -f hex int")

        self.expect("frame variable numbers_list --raw", matching=False,
                    substrs = ['list has 0 items',
                               '{}'])
        self.expect("frame variable &numbers_list._M_impl._M_node --raw", matching=False,
                    substrs = ['list has 0 items',
                               '{}'])

        self.expect("frame variable numbers_list",
                    substrs = ['list has 0 items',
                               '{}'])

        self.expect("p numbers_list",
                    substrs = ['list has 0 items',
                               '{}'])

        self.runCmd("n")

        self.expect("frame variable numbers_list",
                    substrs = ['list has 1 items',
                               '[0] = ',
                               '0x12345678'])

        self.runCmd("n");self.runCmd("n");self.runCmd("n");

        self.expect("frame variable numbers_list",
                    substrs = ['list has 4 items',
                               '[0] = ',
                               '0x12345678',
                               '[1] =',
                               '0x11223344',
                               '[2] =',
                               '0xbeeffeed',
                               '[3] =',
                               '0x00abba00'])

        self.runCmd("n");self.runCmd("n");

        self.expect("frame variable numbers_list",
                    substrs = ['list has 6 items',
                               '[0] = ',
                               '0x12345678',
                               '0x11223344',
                               '0xbeeffeed',
                               '0x00abba00',
                               '[4] =',
                               '0x0abcdef0',
                               '[5] =',
                               '0x0cab0cab'])

        self.expect("p numbers_list",
                    substrs = ['list has 6 items',
                               '[0] = ',
                               '0x12345678',
                               '0x11223344',
                               '0xbeeffeed',
                               '0x00abba00',
                               '[4] =',
                               '0x0abcdef0',
                               '[5] =',
                               '0x0cab0cab'])

        # check access-by-index
        self.expect("frame variable numbers_list[0]",
                    substrs = ['0x12345678']);
        self.expect("frame variable numbers_list[1]",
                    substrs = ['0x11223344']);
        
        # but check that expression does not rely on us
        self.expect("expression numbers_list[0]", matching=False, error=True,
                    substrs = ['0x12345678'])

        # check that MightHaveChildren() gets it right
        self.assertTrue(self.frame().FindVariable("numbers_list").MightHaveChildren(), "numbers_list.MightHaveChildren() says False for non empty!")

        self.runCmd("n")
            
        self.expect("frame variable numbers_list",
                    substrs = ['list has 0 items',
                               '{}'])

        self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n");
            
        self.expect("frame variable numbers_list",
                    substrs = ['list has 4 items',
                               '[0] = ', '1',
                               '[1] = ', '2',
                               '[2] = ', '3',
                               '[3] = ', '4'])            

        self.runCmd("type format delete int")

        self.runCmd("n")
            
        self.expect("frame variable text_list",
            substrs = ['list has 0 items',
                       '{}'])
        
        self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n");

        self.expect("frame variable text_list",
                    substrs = ['list has 4 items',
                               '[0]', 'goofy',
                               '[1]', 'is',
                               '[2]', 'smart',
                               '[3]', '!!!'])

        self.expect("p text_list",
                    substrs = ['list has 4 items',
                               '\"goofy\"',
                               '\"is\"',
                               '\"smart\"',
                               '\"!!!\"'])
        
        # check access-by-index
        self.expect("frame variable text_list[0]",
                    substrs = ['goofy']);
        self.expect("frame variable text_list[3]",
                    substrs = ['!!!']);
        
        # but check that expression does not rely on us
        self.expect("expression text_list[0]", matching=False, error=True,
                    substrs = ['goofy'])

        # check that MightHaveChildren() gets it right
        self.assertTrue(self.frame().FindVariable("text_list").MightHaveChildren(), "text_list.MightHaveChildren() says False for non empty!")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
