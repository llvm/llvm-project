import os
import shutil

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


@skipUnlessDarwin
class TargetModuleAddDsymTest(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    def build_and_hide_dsym(self):
        """Build with a dSYM and move it to where a.out does not find it."""
        self.build(debug_info="dsym")
        hidden = self.getBuildArtifact("hidden")
        os.mkdir(hidden)
        dsym = shutil.move(self.getBuildArtifact("a.out.dSYM"), hidden)
        return os.path.join(dsym, "Contents", "Resources", "DWARF", "a.out")

    def test_no_matching_module(self):
        """A dSYM that matches no module is added as a module."""
        sym_path = self.build_and_hide_dsym()
        target = self.dbg.CreateTarget("")

        self.runCmd("target modules add " + sym_path)

        self.assertEqual(target.GetNumModules(), 1)
        module = target.GetModuleAtIndex(0)
        self.assertEqual(module.GetFileSpec().fullpath, sym_path)
        self.assertEqual(module.FindFunctions("main").GetSize(), 1)

    def test_matching_module(self):
        """A dSYM that matches a module becomes that module's symbol file."""
        sym_path = self.build_and_hide_dsym()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        module = target.GetModuleAtIndex(0)
        self.assertEqual(module.GetSymbolFileSpec().fullpath, exe)
        num_modules = target.GetNumModules()

        self.expect(
            "target modules add " + sym_path,
            substrs=["symbol file", "has been added to", exe],
        )

        self.assertEqual(target.GetNumModules(), num_modules)
        self.assertEqual(module.GetSymbolFileSpec().fullpath, sym_path)
