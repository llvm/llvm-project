""" Testing explicit symbol loading via target symbols add. """
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TargetSymbolsAddCommand(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.source = "main.c"

    @no_debug_info_test  # Prevent the genaration of the dwarf version of this test
    @requireLinux
    def test_target_symbols_add(self):
        """Test that 'target symbols add' can load the symbols
        even if gnu.build-id and gnu_debuglink are not present in the module.
        Similar to test_add_dsym_mid_execution test for macos."""
        self.build()
        exe = self.getBuildArtifact("stripped.out")

        self.target, _, _, _ = lldbutil.run_to_name_breakpoint(
            self, "main", bkpt_module="stripped.out", exe_name="stripped.out"
        )

        exe_module = self.target.GetModuleAtIndex(0)

        # Check that symbols are not loaded and main.c is not know to be
        # the source file.
        self.expect("frame select", substrs=["main.c"], matching=False)

        # Tell LLDB that a.out has symbols for stripped.out
        self.runCmd(
            "target symbols add -s %s %s" % (exe, self.getBuildArtifact("a.out"))
        )

        # Check that symbols are now loaded and main.c is in the output.
        self.expect("frame select", substrs=["main.c"])
