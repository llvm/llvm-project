"""
Test some expressions involving STL data types.
"""

import os, time
import unittest2
import lldb
import lldbutil
from lldbtest import *

class STLTestCase(TestBase):

    mydir = os.path.join("lang", "cpp", "stl")

    # rdar://problem/10400981
    @unittest2.expectedFailure
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test some expressions involving STL data types."""
        self.buildDsym()
        self.step_stl_exprs()

    # rdar://problem/10400981
    @unittest2.expectedFailure
    @dwarf_test
    def test_with_dwarf(self):
        """Test some expressions involving STL data types."""
        self.buildDwarf()
        self.step_stl_exprs()

    @python_api_test
    @dsym_test
    def test_SBType_template_aspects_with_dsym(self):
        """Test APIs for getting template arguments from an SBType."""
        self.buildDsym()
        self.sbtype_template_apis()

    @skipIfGcc # llvm.org/pr15036: crashes during DWARF parsing when built with GCC
    @python_api_test
    @dwarf_test
    def test_SBType_template_aspects_with_dwarf(self):
        """Test APIs for getting template arguments from an SBType."""
        self.buildDwarf()
        self.sbtype_template_apis()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.source = 'main.cpp'
        self.line = line_number(self.source, '// Set break point at this line.')

    def step_stl_exprs(self):
        """Test some expressions involving STL data types."""
        exe = os.path.join(os.getcwd(), "a.out")

        # The following two lines, if uncommented, will enable loggings.
        #self.ci.HandleCommand("log enable -f /tmp/lldb.log lldb default", res)
        #self.assertTrue(res.Succeeded())

        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # rdar://problem/8543077
        # test/stl: clang built binaries results in the breakpoint locations = 3,
        # is this a problem with clang generated debug info?
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Stop at 'std::string hello_world ("Hello World!");'.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['main.cpp:%d' % self.line,
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # Now try some expressions....

        self.runCmd('expr for (int i = 0; i < hello_world.length(); ++i) { (void)printf("%c\\n", hello_world[i]); }')

        # rdar://problem/10373783
        # rdar://problem/10400981
        self.expect('expr associative_array.size()',
            substrs = [' = 3'])
        self.expect('expr associative_array.count(hello_world)',
            substrs = [' = 1'])
        self.expect('expr associative_array[hello_world]',
            substrs = [' = 1'])
        self.expect('expr associative_array["hello"]',
            substrs = [' = 2'])

    def sbtype_template_apis(self):
        """Test APIs for getting template arguments from an SBType."""
        exe = os.path.join(os.getcwd(), 'a.out')

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create the breakpoint inside function 'main'.
        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Get Frame #0.
        self.assertTrue(process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint condition")
        frame0 = thread.GetFrameAtIndex(0)

        # Get the type for variable 'associative_array'.
        associative_array = frame0.FindVariable('associative_array')
        self.DebugSBValue(associative_array)
        self.assertTrue(associative_array, VALID_VARIABLE)
        map_type = associative_array.GetType()
        self.DebugSBType(map_type)
        self.assertTrue(map_type, VALID_TYPE)
        num_template_args = map_type.GetNumberOfTemplateArguments()
        self.assertTrue(num_template_args > 0)

        # We expect the template arguments to contain at least 'string' and 'int'.
        expected_types = { 'string': False, 'int': False }
        for i in range(num_template_args):
            t = map_type.GetTemplateArgumentType(i)
            self.DebugSBType(t)
            self.assertTrue(t, VALID_TYPE)
            name = t.GetName()
            if 'string' in name:
                expected_types['string'] = True
            elif 'int' == name:
                expected_types['int'] = True

        # Check that both entries of the dictionary have 'True' as the value.
        self.assertTrue(all(expected_types.values()))


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
