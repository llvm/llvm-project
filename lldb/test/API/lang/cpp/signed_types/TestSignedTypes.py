"""
Test that variables with signed types display correctly.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class SignedTypesTestCase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.source = 'main.cpp'
        self.line = line_number(
            self.source, '// Set break point at this line.')

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24489")
    def test(self):
        """Test that variables with signed types display correctly."""
        self.build()

        # Run in synchronous mode
        self.dbg.SetAsync(False)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        lldbutil.run_break_set_by_file_and_line(
            self, self.source, self.line, num_expected_locations=1, loc_exact=True)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno = 1, expected_hit_count = 1)

        # Execute the puts().
        self.runCmd("thread step-over")

        # Test that signed types display correctly.
        self.expect(
            "frame variable --show-types --no-args",
            VARIABLES_DISPLAYED_CORRECTLY,
            patterns=[
                "\((short int|short)\) the_signed_short = 99",
                "\((signed char|char)\) the_signed_char = 'c'"],
            substrs=[
                "(int) the_signed_int = 99",
                "(long) the_signed_long = 99",
                "(long long) the_signed_long_long = 99"])
