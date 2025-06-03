"""Test that forward declaration of a c++ template gets resolved correctly."""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class ForwardDeclarationTestCase(TestBase):
    def do_test(self, dictionary=None):
        """Display *bar_ptr when stopped on a function with forward declaration of struct bar."""
        self.build(dictionary=dictionary)
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        environment = self.registerSharedLibrariesWithTarget(target, ["foo"])

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_break_set_by_symbol(self, "foo", num_expected_locations=1)

        process = target.LaunchSimple(
            None, environment, self.get_process_working_directory()
        )
        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno=1, expected_hit_count=1)

        self.expect_expr(
            "*bar_ptr",
            result_type="bar<int>",
            result_children=[ValueCheck(value="47", name="a", type="int")],
        )

    def test(self):
        self.do_test()

    @no_debug_info_test
    @skipIfDarwin
    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler_version=["<", "8.0"])
    @expectedFailureAll(oslist=["windows"])
    def test_debug_names(self):
        """Test that we are able to find complete types when using DWARF v5
        accelerator tables"""
        self.do_test(dict(CFLAGS_EXTRAS="-gdwarf-5 -gpubnames"))

    @no_debug_info_test
    @skipIf(compiler=no_match("clang"))
    def test_simple_template_names(self):
        """Test that we are able to find complete types when using DWARF v5
        accelerator tables"""
        self.do_test(dict(CFLAGS_EXTRAS="-gsimple-template-names"))
