"""Test that the 'add-dsym', aka 'target symbols add', succeeds in the middle of debug session."""


from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@requireDarwin
class AddDsymMidExecutionCommandCase(TestBase):
    SHARED_BUILD_TESTCASE = False

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.source = "main.c"

    @no_debug_info_test  # Prevent the genaration of the dwarf version of this test
    def test_add_dsym_mid_execution(self):
        """Test that add-dsym mid-execution loads the symbols at the right place for a slid binary."""
        self.build(debug_info="dsym")

        self.runCmd("settings set target.disable-aslr false")
        self.target, self.process, _, _ = lldbutil.run_to_name_breakpoint(
            self, "main", bkpt_module="a.out"
        )

        self.runCmd("add-dsym " + self.getBuildArtifact("hide.app/Contents/a.out.dSYM"))

        self.expect("frame select", substrs=["a.out`main at main.c"])
