"""
Test that we can hit breakpoints in global constructors
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfTargetDoesNotSupportSharedLibraries()
class TestBreakpointInGlobalConstructors(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()
        self.line_foo = line_number("foo.cpp", "// !BR_foo")
        self.line_main = line_number("main.cpp", "// !BR_main")

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        env = self.registerSharedLibrariesWithTarget(target, ["foo"])

        bp_main = lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line_main
        )

        bp_foo = lldbutil.run_break_set_by_file_and_line(
            self, "foo.cpp", self.line_foo, num_expected_locations=-2
        )

        launch_info = target.GetLaunchInfo()
        launch_info.SetEnvironmentEntries(env, False)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        lldbutil.run_to_breakpoint_do_run(
            self, target, target.FindBreakpointByID(bp_foo), launch_info
        )

        self.runCmd("continue")

        self.assertIsNotNone(
            lldbutil.get_one_thread_stopped_at_breakpoint_id(self.process(), bp_main)
        )
