"""
Test the 'memory find' command.
"""



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *


class MemoryFindTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.cpp', '// break here')

    def test_memory_find(self):
        """Test the 'memory find' command."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break in main() after the variables are assigned values.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        # Test the memory find commands.

        self.expect(
            'memory find -s "in const" `stringdata` `stringdata+(int)strlen(stringdata)`',
            substrs=[
                'data found at location: 0x',
                '69 6e 20 63',
                'in const'])

        self.expect(
            'memory find -e "(uint8_t)0x22" `&bytedata[0]` `&bytedata[15]`',
            substrs=[
                'data found at location: 0x',
                '22 33 44 55 66'])

        self.expect(
            'memory find -e "(uint8_t)0x22" `&bytedata[0]` `&bytedata[2]`',
            substrs=['data not found within the range.'])

        self.expect('memory find -s "nothere" `stringdata` `stringdata+5`',
                    substrs=['data not found within the range.'])

        self.expect('memory find -s "nothere" `stringdata` `stringdata+10`',
                    substrs=['data not found within the range.'])
