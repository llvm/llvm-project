"""Test that forward declaration of a c++ template gets resolved correctly."""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


@skipIfTargetDoesNotSupportSharedLibraries()
class ForwardDeclarationTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False

    def do_test(self, dictionary=None):
        """Display *bar_ptr when stopped on a function with forward declaration of struct bar."""
        self.build(dictionary=dictionary)
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        environment = self.registerSharedLibrariesWithTarget(target, ["foo"])

        # Break inside the foo function which takes a bar_ptr argument.
        bkpt_id = lldbutil.run_break_set_by_symbol(
            self, "foo", num_expected_locations=1
        )

        launch_info = target.GetLaunchInfo()
        launch_info.SetEnvironmentEntries(environment, False)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        lldbutil.run_to_breakpoint_do_run(
            self, target, target.FindBreakpointByID(bkpt_id), launch_info
        )

        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno=1, expected_hit_count=1)

        self.expect_expr(
            "*bar_ptr",
            result_type="bar<int>",
            result_children=[ValueCheck(value="47", name="a", type="int")],
        )

    @expectedFailureAll(debug_info=["pdb"], bugnumber="llvm.org/pr149498")
    def test(self):
        self.do_test()

    @no_debug_info_test
    @skipIfDarwin
    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler_version=["<", "8.0"])
    def test_debug_names(self):
        """Test that we are able to find complete types when using DWARF v5
        accelerator tables"""
        self.do_test(dict(CFLAGS_EXTRAS="-gdwarf-5 -gpubnames"))

    @no_debug_info_test
    @requireClang
    def test_simple_template_names(self):
        """Test that we are able to find complete types when using DWARF v5
        accelerator tables"""
        self.do_test(dict(CFLAGS_EXTRAS="-gsimple-template-names"))
