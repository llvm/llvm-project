"""Show bitfields and check that they display correctly."""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class BitfieldsTestCase(TestBase):

    mydir = os.path.join("lang", "c", "bitfields")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test 'frame variable ...' on a variable with bitfields."""
        self.buildDsym()
        self.bitfields_variable()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_with_dsym_and_python_api(self):
        """Use Python APIs to inspect a bitfields variable."""
        self.buildDsym()
        self.bitfields_variable_python()

    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Test 'frame variable ...' on a variable with bitfields."""
        self.buildDwarf()
        self.bitfields_variable()

    @python_api_test
    @dwarf_test
    def test_with_dwarf_and_python_api(self):
        """Use Python APIs to inspect a bitfields variable."""
        self.buildDwarf()
        self.bitfields_variable_python()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    def bitfields_variable(self):
        """Test 'frame variable ...' on a variable with bitfields."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # This should display correctly.
        self.expect("frame variable -T bits", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(uint32_t:1) b1 = 1',
                       '(uint32_t:2) b2 = 3',
                       '(uint32_t:3) b3 = 7',
                       '(uint32_t) b4 = 15',
                       '(uint32_t:5) b5 = 31',
                       '(uint32_t:6) b6 = 63',
                       '(uint32_t:7) b7 = 127',
                       '(uint32_t:4) four = 15'])

        # And so should this.
        # rdar://problem/8348251
        self.expect("frame variable -T", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(uint32_t:1) b1 = 1',
                       '(uint32_t:2) b2 = 3',
                       '(uint32_t:3) b3 = 7',
                       '(uint32_t) b4 = 15',
                       '(uint32_t:5) b5 = 31',
                       '(uint32_t:6) b6 = 63',
                       '(uint32_t:7) b7 = 127',
                       '(uint32_t:4) four = 15'])

    def bitfields_variable_python(self):
        """Use Python APIs to inspect a bitfields variable."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.c", self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        thread = target.GetProcess().GetThreadAtIndex(0)
        if thread.GetStopReason() != lldb.eStopReasonBreakpoint:
            from lldbutil import stop_reason_to_str
            self.fail(STOPPED_DUE_TO_BREAKPOINT_WITH_STOP_REASON_AS %
                      stop_reason_to_str(thread.GetStopReason()))

        # The breakpoint should have a hit count of 1.
        self.assertTrue(breakpoint.GetHitCount() == 1, BREAKPOINT_HIT_ONCE)

        # Lookup the "bits" variable which contains 8 bitfields.
        frame = thread.GetFrameAtIndex(0)
        bits = frame.FindVariable("bits")
        self.DebugSBValue(bits)
        self.assertTrue(bits.GetTypeName() == 'Bits', "bits.GetTypeName() == 'Bits'");
        self.assertTrue(bits.GetNumChildren() == 10, "bits.GetNumChildren() == 10");
        self.assertTrue(bits.GetByteSize() == 32, "bits.GetByteSize() == 32");

        # Notice the pattern of int(b1.GetValue(), 0).  We pass a base of 0
        # so that the proper radix is determined based on the contents of the
        # string.
        b1 = bits.GetChildMemberWithName("b1")
        self.DebugSBValue(b1)
        self.assertTrue(b1.GetName() == "b1" and
                        b1.GetTypeName() == "uint32_t:1" and
                        b1.IsInScope() and
                        int(b1.GetValue(), 0) == 1,
                        'bits.b1 has type uint32_t:1, is in scope, and == 1')

        b7 = bits.GetChildMemberWithName("b7")
        self.DebugSBValue(b7)
        self.assertTrue(b7.GetName() == "b7" and
                        b7.GetTypeName() == "uint32_t:7" and
                        b7.IsInScope() and
                        int(b7.GetValue(), 0) == 127,
                        'bits.b7 has type uint32_t:7, is in scope, and == 127')

        four = bits.GetChildMemberWithName("four")
        self.DebugSBValue(four)
        self.assertTrue(four.GetName() == "four" and
                        four.GetTypeName() == "uint32_t:4" and
                        four.IsInScope() and
                        int(four.GetValue(), 0) == 15,
                        'bits.four has type uint32_t:4, is in scope, and == 15')

        # Now kill the process, and we are done.
        rc = target.GetProcess().Kill()
        self.assertTrue(rc.Success())


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
