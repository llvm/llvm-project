"""Test that types defined in shared libraries with stripped symbols work correctly."""


import unittest
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfTargetDoesNotSupportSharedLibraries()
class SharedLibStrippedTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False

    @expectedFailureAll(oslist=["windows"])
    # Sometimes fails with:
    # error: Couldn't allocate space for materialized struct: Couldn't malloc: address space is full
    # On 32 bit Arm Linux.
    @skipIf(archs=["arm$"])
    def test_expr(self):
        """Test that types work when defined in a shared library and forward-declared in the main executable"""
        if "clang" in self.getCompiler() and "3.4" in self.getCompilerVersion():
            self.skipTest(
                "llvm.org/pr16214 -- clang emits partial DWARF for structures referenced via typedef"
            )

        self.build()
        self.common_setup()

        # This should display correctly.
        self.expect(
            "expression --show-types -- *my_foo_ptr",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["(foo)", "(sub_foo)", "other_element = 3"],
        )

    @expectedFailureAll(oslist=["windows"])
    @unittest.expectedFailure  # llvm.org/PR36712
    def test_frame_variable(self):
        """Test that types work when defined in a shared library and forward-declared in the main executable"""
        self.build()
        self.common_setup()

        # This should display correctly.
        self.expect(
            "frame variable --show-types -- *my_foo_ptr",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["(foo)", "(sub_foo)", "other_element = 3"],
        )

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.source = "main.c"
        self.line = line_number(self.source, "// Set breakpoint 0 here.")
        self.shlib_names = ["foo"]

    def common_setup(self):
        # Run in synchronous mode
        self.dbg.SetAsync(False)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        # Break inside the foo function which takes a bar_ptr argument.
        bkpt_id = lldbutil.run_break_set_by_file_and_line(
            self, self.source, self.line, num_expected_locations=1, loc_exact=True
        )

        # Register our shared libraries for remote targets so they get
        # automatically uploaded
        environment = self.registerSharedLibrariesWithTarget(target, self.shlib_names)

        launch_info = target.GetLaunchInfo()
        launch_info.SetEnvironmentEntries(environment, False)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        lldbutil.run_to_breakpoint_do_run(
            self, target, target.FindBreakpointByID(bkpt_id), launch_info
        )

        lldbutil.check_breakpoint(self, bpno=1, expected_hit_count=1)
