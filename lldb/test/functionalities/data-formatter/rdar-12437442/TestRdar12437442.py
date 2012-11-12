"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import datetime
import lldbutil

class DataFormatterRdar12437442TestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "rdar-12437442")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_rdar12437442_with_dsym_and_run_command(self):
        """Test that we update SBValues correctly as dynamic types change."""
        self.buildDsym()
        self.rdar12437442_tester()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_rdar12437442_with_dwarf_and_run_command(self):
        """Test that we update SBValues correctly as dynamic types change."""
        self.buildDwarf()
        self.rdar12437442_tester()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.m', '// Set break point at this line.')

    def rdar12437442_tester(self):
        """Test that we update SBValues correctly as dynamic types change."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

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
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("log enable lldb types -f types.log")

        # Now run the bulk of the test
        id_x = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable("x")
        id_x.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        id_x.SetPreferSyntheticValue(True)
        
        if self.TraceOn():
            self.runCmd("frame variable x -d run-target --ptr-depth 1")
        
        self.assertTrue(id_x.GetSummary() == '@"5 objects"', "array does not get correct summary")

        self.runCmd("next")

        id_x = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable("x")
        id_x.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        id_x.SetPreferSyntheticValue(True)

        if self.TraceOn():
            self.runCmd("frame variable x -d run-target --ptr-depth 1")

        self.assertTrue(id_x.GetNumChildren() == 7, "dictionary does not have 7 children")
        id_x.SetPreferSyntheticValue(False)
        self.assertFalse(id_x.GetNumChildren() == 7, "dictionary still looks synthetic")
        id_x.SetPreferSyntheticValue(True)
        self.assertTrue(id_x.GetSummary() == "7 key/value pairs", "dictionary does not get correct summary")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
