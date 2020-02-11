"""Test variable lookup when stopped in inline functions."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class InlinesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number(
            'inlines.cpp',
            '// Set break point at this line.')

    @expectedFailureAll("llvm.org/pr26710", oslist=["linux"], compiler="gcc")
    def test(self):
        """Test that local variables are visible in expressions."""
        self.build()
        self.runToBreakpoint()

        # Check that 'frame variable' finds a variable
        self.expect(
            "frame variable inner_input",
            VARIABLES_DISPLAYED_CORRECTLY,
            startstr='(int) inner_input =')

        # Check that 'expr' finds a variable
        self.expect("expr inner_input", VARIABLES_DISPLAYED_CORRECTLY,
                    startstr='(int) $0 =')

    def runToBreakpoint(self):
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        lldbutil.run_break_set_by_file_and_line(
            self,
            "inlines.cpp",
            self.line,
            num_expected_locations=2,
            loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])
