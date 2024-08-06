import os
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


@skipUnlessDarwin
class TargetModuleAddDsymTest(TestBase):
    @no_debug_info_test
    def test_target_module_add(self):
        """Test that you can add a dSYM as a module."""
        self.build(debug_info="dsym")

        exe_path = self.getBuildArtifact("a.out")
        dsym_path = exe_path + ".dSYM"
        sym_path = os.path.join(dsym_path, "Contents", "Resources", "DWARF", "a.out")

        exe = self.getBuildArtifact("a.out")
        self.dbg.CreateTarget(exe)

        self.runCmd("target module add %s" % sym_path)
        self.expect("image list", substrs=[sym_path])
