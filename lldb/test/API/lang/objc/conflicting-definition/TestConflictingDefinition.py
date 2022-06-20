"""Test that types defined in shared libraries work correctly."""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestRealDefinition(TestBase):

    def test_frame_var_after_stop_at_implementation(self):
        """Test that we can find the implementation for an objective C type"""
        if self.getArchitecture() == 'i386':
            self.skipTest("requires modern objc runtime")
        self.build()
        self.shlib_names = ["libTestExt.dylib", "libTest.dylib"]
        self.common_setup()

        line = line_number('TestExt/TestExt.m', '// break here')
        lldbutil.run_break_set_by_file_and_line(
            self, 'TestExt.m', line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        lldbutil.check_breakpoint(self, bpno = 1, expected_hit_count = 1)

        # This should display correctly.
        self.expect(
            "expr 42",
            "A simple expression should execute correctly",
            substrs=[
                "42"])

    def common_setup(self):
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.registerSharedLibrariesWithTarget(target, self.shlib_names)

        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
