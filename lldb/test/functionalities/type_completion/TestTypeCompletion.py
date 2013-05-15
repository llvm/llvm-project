"""
Check that types only get completed when necessary.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class TypeCompletionTestCase(TestBase):

    mydir = os.path.join("functionalities", "type_completion")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Check that types only get completed when necessary."""
        self.buildDsym()
        self.type_completion_commands()

    @dwarf_test
    @expectedFailureGcc # llvm.org/pr15301 LLDB prints incorrect sizes of STL containers
    def test_with_dwarf_and_run_command(self):
        """Check that types only get completed when necessary."""
        self.buildDwarf()
        self.type_completion_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def type_completion_commands(self):
        """Check that types only get completed when necessary."""
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
            self.runCmd('type category enable gnu-libstdc++', check=False)
            self.runCmd('type category enable libcxx', check=False)

        self.runCmd('type category disable gnu-libstdc++', check=False)
        self.runCmd('type category disable libcxx', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        p_vector = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable('p')
        p_type = p_vector.GetType()
        self.assertFalse(p_type.IsTypeComplete(), 'vector<T> complete but it should not be')

        self.runCmd("next")
        self.runCmd("next")

        p_vector = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable('p')
        p_type = p_vector.GetType()
        self.assertFalse(p_type.IsTypeComplete(), 'vector<T> complete but it should not be')

        self.runCmd("next")
        self.runCmd("next")

        self.runCmd("frame variable p --show-types")

        p_vector = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable('p')
        p_type = p_vector.GetType()
        self.assertTrue(p_type.IsTypeComplete(), 'vector<T> should now be complete')
        name_address_type = p_type.GetTemplateArgumentType(0)
        self.assertTrue(name_address_type.IsValid(), 'NameAndAddress should be valid')
        self.assertFalse(name_address_type.IsTypeComplete(), 'NameAndAddress complete but it should not be')

        self.runCmd("next")
        self.runCmd("next")

        self.runCmd("frame variable guy --show-types")

        p_vector = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable('p')
        p_type = p_vector.GetType()
        self.assertTrue(p_type.IsTypeComplete(), 'vector<T> should now be complete')
        name_address_type = p_type.GetTemplateArgumentType(0)
        self.assertTrue(name_address_type.IsValid(), 'NameAndAddress should be valid')
        self.assertTrue(name_address_type.IsTypeComplete(), 'NameAndAddress should now be complete')
        field0 = name_address_type.GetFieldAtIndex(0)
        if self.TraceOn():
             print 'field0: ' + str(field0)
        self.assertTrue(field0.IsValid(), 'NameAndAddress::m_name should be valid')
        string = field0.GetType().GetPointeeType()
        if self.TraceOn():
             print 'string: ' + str(string)
        self.assertTrue(string.IsValid(), 'std::string should be valid')
        self.assertFalse(string.IsTypeComplete(), 'std::string complete but it should not be')

        self.runCmd("next")
        self.runCmd("next")

        p_vector = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable('p')
        p_type = p_vector.GetType()
        self.assertTrue(p_type.IsTypeComplete(), 'vector<T> should now be complete')
        name_address_type = p_type.GetTemplateArgumentType(0)
        self.assertTrue(name_address_type.IsValid(), 'NameAndAddress should be valid')
        self.assertTrue(name_address_type.IsTypeComplete(), 'NameAndAddress should now be complete')
        field0 = name_address_type.GetFieldAtIndex(0)
        if self.TraceOn():
             print 'field0: ' + str(field0)
        self.assertTrue(field0.IsValid(), 'NameAndAddress::m_name should be valid')
        string = field0.GetType().GetPointeeType()
        if self.TraceOn():
             print 'string: ' + str(string)
        self.assertTrue(string.IsValid(), 'std::string should be valid')
        self.assertFalse(string.IsTypeComplete(), 'std::string complete but it should not be')

        self.runCmd('type category enable gnu-libstdc++', check=False)
        self.runCmd('type category enable libcxx', check=False)
        self.runCmd('frame variable guy --show-types')

        p_vector = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable('p')
        p_type = p_vector.GetType()
        self.assertTrue(p_type.IsTypeComplete(), 'vector<T> should now be complete')
        name_address_type = p_type.GetTemplateArgumentType(0)
        self.assertTrue(name_address_type.IsValid(), 'NameAndAddress should be valid')
        self.assertTrue(name_address_type.IsTypeComplete(), 'NameAndAddress should now be complete')
        field0 = name_address_type.GetFieldAtIndex(0)
        self.assertTrue(field0.IsValid(), 'NameAndAddress::m_name should be valid')
        string = field0.GetType().GetPointeeType()
        self.assertTrue(string.IsValid(), 'std::string should be valid')
        self.assertTrue(string.IsTypeComplete(), 'std::string should now be complete')

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
