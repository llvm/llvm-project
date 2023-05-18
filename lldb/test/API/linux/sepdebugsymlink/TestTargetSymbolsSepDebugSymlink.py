""" Testing separate debug info loading for base binary with a symlink. """
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestTargetSymbolsSepDebugSymlink(TestBase):
    @no_debug_info_test  # Prevent the genaration of the dwarf version of this test
    @skipUnlessPlatform(["linux"])
    @skipIf(hostoslist=["windows"])
    @skipIfRemote  # llvm.org/pr36237
    def test_target_symbols_sepdebug_symlink_case(self):
        self.build()
        exe = self.getBuildArtifact("dirsymlink/stripped.symlink")

        lldbutil.run_to_name_breakpoint(self, "main", exe_name=exe)
