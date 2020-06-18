"""Test that forward declaration of a data structure gets resolved correctly."""



import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class ForwardDeclarationTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def do_test(self, dictionary=None):
        """Display *bar_ptr when stopped on a function with forward declaration of struct bar."""
        self.build(dictionary=dictionary)
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_break_set_by_symbol(
            self, "foo", num_expected_locations=1, sym_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        # This should display correctly.
        # Note that the member fields of a = 1 and b = 2 is by design.
        self.expect(
            "frame variable --show-types *bar_ptr",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                '(bar) *bar_ptr = ',
                '(int) a = 1',
                '(int) b = 2'])

        # And so should this.
        self.expect(
            "expression --show-types -- *bar_ptr",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                '(bar)',
                '(int) a = 1',
                '(int) b = 2'])

    def test(self):
        self.do_test()

    @no_debug_info_test
    @skipIfDarwin
    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler_version=["<", "7.0"])
    @expectedFailureAll(oslist=["windows"])
    def test_debug_names(self):
        """Test that we are able to find complete types when using DWARF v5
        accelerator tables"""
        self.do_test(dict(CFLAGS_EXTRAS="-gdwarf-5 -gpubnames"))
