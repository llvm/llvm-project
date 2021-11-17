"""Show local variables and check that they can be inspected.

This test was added after we made a change in clang to normalize
DW_OP_constu(X < 32) to DW_OP_litX which broke the debugger because
it didn't read the value as an unsigned.
"""



from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LocalVariablesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.source = 'main.c'
        self.line = line_number(
            self.source, '// Set break point at this line.')

    @skipIfWindows
    def test_c_local_variables(self):
        """Test local variable value."""
        self.build()

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        # Break inside the main.
        lldbutil.run_break_set_by_file_and_line(
            self, self.source, self.line, num_expected_locations=1, loc_exact=True)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno = 1, expected_hit_count = 1)

        self.expect("frame variable i", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['(unsigned int) i = 10'])
