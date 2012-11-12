"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class LibcxxMapDataFormatterTestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "data-formatter-stl", "libcxx", "map")

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

        self.expect('image list',substrs=['libc++.1.dylib','libc++abi.dylib'])

        self.runCmd("frame variable ii -T")
        
        self.runCmd("type summary add -x \"std::__1::map<\" --summary-string \"map has ${svar%#} items\" -e") 
        
        self.expect('frame variable ii',
            substrs = ['map has 0 items',
                       '{}'])

        self.runCmd("n");self.runCmd("n");

        self.expect('frame variable ii',
                    substrs = ['map has 2 items',
                               '[0] = {',
                               'first = 0',
                               'second = 0',
                               '[1] = {',
                               'first = 1',
                               'second = 1'])

        self.runCmd("n");self.runCmd("n");

        self.expect('frame variable ii',
                    substrs = ['map has 4 items',
                               '[2] = {',
                               'first = 2',
                               'second = 0',
                               '[3] = {',
                               'first = 3',
                               'second = 1'])

        self.runCmd("n");self.runCmd("n");
        self.runCmd("n");self.runCmd("n");
        self.runCmd("frame select 0")

        self.expect("frame variable ii",
                    substrs = ['map has 8 items',
                               '[5] = {',
                               'first = 5',
                               'second = 0',
                               '[7] = {',
                               'first = 7',
                               'second = 1'])

        self.expect("p ii",
                    substrs = ['map has 8 items',
                               '[5] = {',
                               'first = 5',
                               'second = 0',
                               '[7] = {',
                               'first = 7',
                               'second = 1'])

        # check access-by-index
        self.expect("frame variable ii[0]",
                    substrs = ['first = 0',
                               'second = 0']);
        self.expect("frame variable ii[3]",
                    substrs = ['first =',
                               'second =']);

        # check that MightHaveChildren() gets it right
        self.assertTrue(self.frame().FindVariable("ii").MightHaveChildren(), "ii.MightHaveChildren() says False for non empty!")

        # check that the expression parser does not make use of
        # synthetic children instead of running code
        # TOT clang has a fix for this, which makes the expression command here succeed
        # since this would make the test fail or succeed depending on clang version in use
        # this is safer commented for the time being
        #self.expect("expression ii[8]", matching=False, error=True,
        #            substrs = ['1234567'])

        self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd('frame select 0')
        
        self.expect('frame variable ii',
                    substrs = ['map has 0 items',
                               '{}'])
        
        self.runCmd("frame variable si -T")

        self.expect('frame variable si',
                    substrs = ['map has 0 items',
                               '{}'])

        self.runCmd("n");self.runCmd("n");self.runCmd('frame select 0')

        self.expect('frame variable si',
                    substrs = ['map has 1 items',
                               '[0] = ',
                               'first = \"zero\"',
                               'second = 0'])

        self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n");
        self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n")
        self.runCmd("n");self.runCmd("n");self.runCmd('frame select 0')

        self.expect("frame variable si",
                    substrs = ['map has 4 items',
                               '[0] = ',
                               'first = \"zero\"',
                               'second = 0',
                                '[1] = ',
                                'first = \"one\"',
                                'second = 1',
                                '[2] = ',
                                'first = \"two\"',
                                'second = 2',
                                '[3] = ',
                                'first = \"three\"',
                                'second = 3'])

        self.expect("p si",
                    substrs = ['map has 4 items',
                               '[0] = ',
                               'first = \"zero\"',
                               'second = 0',
                               '[1] = ',
                               'first = \"one\"',
                               'second = 1',
                               '[2] = ',
                               'first = \"two\"',
                               'second = 2',
                               '[3] = ',
                               'first = \"three\"',
                               'second = 3'])

        # check that MightHaveChildren() gets it right
        self.assertTrue(self.frame().FindVariable("si").MightHaveChildren(), "si.MightHaveChildren() says False for non empty!")

        # check access-by-index
        self.expect("frame variable si[0]",
                    substrs = ['first = ', 'one',
                               'second = 1']);
        
        # check that the expression parser does not make use of
        # synthetic children instead of running code
        # TOT clang has a fix for this, which makes the expression command here succeed
        # since this would make the test fail or succeed depending on clang version in use
        # this is safer commented for the time being
        #self.expect("expression si[0]", matching=False, error=True,
        #            substrs = ['first = ', 'zero'])

        self.runCmd("n");self.runCmd("n");
        
        self.expect('frame variable si',
                    substrs = ['map has 0 items',
                               '{}'])

        self.runCmd("n")
        self.runCmd("frame variable is -T")
        
        self.expect('frame variable is',
                    substrs = ['map has 0 items',
                               '{}'])

        self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n");

        self.expect("frame variable is",
                    substrs = ['map has 4 items',
                               '[0] = ',
                               'second = \"goofy\"',
                               'first = 85',
                               '[1] = ',
                               'second = \"is\"',
                               'first = 1',
                               '[2] = ',
                               'second = \"smart\"',
                               'first = 2',
                               '[3] = ',
                               'second = \"!!!\"',
                               'first = 3'])
        
        self.expect("p is",
                    substrs = ['map has 4 items',
                               '[0] = ',
                               'second = \"goofy\"',
                               'first = 85',
                               '[1] = ',
                               'second = \"is\"',
                               'first = 1',
                               '[2] = ',
                               'second = \"smart\"',
                               'first = 2',
                               '[3] = ',
                               'second = \"!!!\"',
                               'first = 3'])

        # check that MightHaveChildren() gets it right
        self.assertTrue(self.frame().FindVariable("is").MightHaveChildren(), "is.MightHaveChildren() says False for non empty!")

        # check access-by-index
        self.expect("frame variable is[0]",
                    substrs = ['first = ',
                               'second =']);
        
        # check that the expression parser does not make use of
        # synthetic children instead of running code
        # TOT clang has a fix for this, which makes the expression command here succeed
        # since this would make the test fail or succeed depending on clang version in use
        # this is safer commented for the time being
        #self.expect("expression is[0]", matching=False, error=True,
        #            substrs = ['first = ', 'goofy'])

        self.runCmd("n")
        
        self.expect('frame variable is',
                    substrs = ['map has 0 items',
                               '{}'])

        self.runCmd("n");self.runCmd("n");
        self.runCmd("frame variable ss -T")
        
        self.expect('frame variable ss',
                    substrs = ['map has 0 items',
                               '{}'])

        self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n");
        self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd('frame select 0')

        self.expect("frame variable ss",
                    substrs = ['map has 3 items',
                               '[0] = ',
                               'second = \"hello\"',
                               'first = \"ciao\"',
                               '[1] = ',
                               'second = \"house\"',
                               'first = \"casa\"',
                               '[2] = ',
                               'second = \"cat\"',
                               'first = \"gatto\"'])
        
        self.expect("p ss",
                    substrs = ['map has 3 items',
                               '[0] = ',
                               'second = \"hello\"',
                               'first = \"ciao\"',
                               '[1] = ',
                               'second = \"house\"',
                               'first = \"casa\"',
                               '[2] = ',
                               'second = \"cat\"',
                               'first = \"gatto\"'])

        # check that MightHaveChildren() gets it right
        self.assertTrue(self.frame().FindVariable("ss").MightHaveChildren(), "ss.MightHaveChildren() says False for non empty!")

        # check access-by-index
        self.expect("frame variable ss[2]",
                    substrs = ['gatto', 'cat']);
        
        # check that the expression parser does not make use of
        # synthetic children instead of running code
        # TOT clang has a fix for this, which makes the expression command here succeed
        # since this would make the test fail or succeed depending on clang version in use
        # this is safer commented for the time being
        #self.expect("expression ss[3]", matching=False, error=True,
        #            substrs = ['gatto'])

        self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd('frame select 0')
        
        self.expect('frame variable ss',
                    substrs = ['map has 0 items',
                               '{}'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
