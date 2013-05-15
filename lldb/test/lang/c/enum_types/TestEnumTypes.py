"""Look up enum type information and check for correct display."""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class EnumTypesTestCase(TestBase):

    mydir = os.path.join("lang", "c", "enum_types")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test 'image lookup -t days' and check for correct display and enum value printing."""
        self.buildDsym()
        self.image_lookup_for_enum_type()

    # rdar://problem/8394746
    # 'image lookup -t days' returns nothing with dwarf debug format.
    @dwarf_test
    def test_with_dwarf(self):
        """Test 'image lookup -t days' and check for correct display and enum value printing."""
        self.buildDwarf()
        self.image_lookup_for_enum_type()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    def image_lookup_for_enum_type(self):
        """Test 'image lookup -t days' and check for correct display."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        bkpt_id = lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # Look up information about the 'days' enum type.
        # Check for correct display.
        self.expect("image lookup -t days", DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs = ['enum days {',
                       'Monday',
                       'Tuesday',
                       'Wednesday',
                       'Thursday',
                       'Friday',
                       'Saturday',
                       'Sunday',
                       'kNumDays',
                       '}'])

        enum_values = [ '-4', 
                        'Monday', 
                        'Tuesday', 
                        'Wednesday', 
                        'Thursday',
                        'Friday',
                        'Saturday',
                        'Sunday',
                        'kNumDays',
                        '5'];

        bkpt = self.target().FindBreakpointByID(bkpt_id)
        for enum_value in enum_values:
            self.expect("frame variable day", 'check for valid enumeration value',
                substrs = [enum_value])
            lldbutil.continue_to_breakpoint (self.process(), bkpt)
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
