"""
Check if changing Format on an SBValue correctly propagates that new format to children as it should
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class FormatPropagationTestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "format-propagation")

    # rdar://problem/14035604
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Check if changing Format on an SBValue correctly propagates that new format to children as it should"""
        self.buildDsym()
        self.propagate_test_commands()

    # rdar://problem/14035604
    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Check if changing Format on an SBValue correctly propagates that new format to children as it should"""
        self.buildDwarf()
        self.propagate_test_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def propagate_test_commands(self):
        """Check for an issue where capping does not work because the Target pointer appears to be changing behind our backs."""
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
            pass

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # extract the parent and the children
        frame = self.frame()
        parent = self.frame().FindVariable("f")
        self.assertTrue(parent != None and parent.IsValid(),"could not find f")
        X = parent.GetChildMemberWithName("X")
        self.assertTrue(X != None and X.IsValid(),"could not find X")
        Y = parent.GetChildMemberWithName("Y")
        self.assertTrue(Y != None and Y.IsValid(),"could not find Y")
        # check their values now
        self.assertTrue(X.GetValue() == "1", "X has an invalid value")
        self.assertTrue(Y.GetValue() == "2", "Y has an invalid value")
        # set the format on the parent
        parent.SetFormat(lldb.eFormatHex)
        self.assertTrue(X.GetValue() == "0x00000001", "X has not changed format")
        self.assertTrue(Y.GetValue() == "0x00000002", "Y has not changed format")
        # Step and check if the values make sense still
        self.runCmd("next")
        self.assertTrue(X.GetValue() == "0x00000004", "X has not become 4")
        self.assertTrue(Y.GetValue() == "0x00000002", "Y has not stuck as hex")
        # Check that children can still make their own choices
        Y.SetFormat(lldb.eFormatDecimal)
        self.assertTrue(X.GetValue() == "0x00000004", "X is still hex")
        self.assertTrue(Y.GetValue() == "2", "Y has not been reset")
        # Make a few more changes
        parent.SetFormat(lldb.eFormatDefault)
        X.SetFormat(lldb.eFormatHex)
        Y.SetFormat(lldb.eFormatDefault)
        self.assertTrue(X.GetValue() == "0x00000004", "X is not hex as it asked")
        self.assertTrue(Y.GetValue() == "2", "Y is not defaulted")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
