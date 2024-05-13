"""Test that the 'add-dsym', aka 'target symbols add', succeeds in the middle of debug session."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipUnlessDarwin
class AddDsymMidExecutionCommandCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.source = "main.c"

    @no_debug_info_test  # Prevent the genaration of the dwarf version of this test
    def test_add_dsym_mid_execution(self):
        """Test that add-dsym mid-execution loads the symbols at the right place for a slid binary."""
        self.build(debug_info="dsym")
        exe = self.getBuildArtifact("a.out")

        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)

        main_bp = self.target.BreakpointCreateByName("main", "a.out")
        self.assertTrue(main_bp, VALID_BREAKPOINT)

        self.runCmd("settings set target.disable-aslr false")
        self.process = self.target.LaunchSimple(
            None, None, self.get_process_working_directory()
        )
        self.assertTrue(self.process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.assertState(
            self.process.GetState(), lldb.eStateStopped, STOPPED_DUE_TO_BREAKPOINT
        )

        self.runCmd("add-dsym " + self.getBuildArtifact("hide.app/Contents/a.out.dSYM"))

        self.expect("frame select", substrs=["a.out`main at main.c"])
