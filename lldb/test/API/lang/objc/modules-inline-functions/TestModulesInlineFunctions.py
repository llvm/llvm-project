"""Test that inline functions from modules are imported correctly"""




import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ModulesInlineFunctionsTestCase(TestBase):

    @add_test_categories(["gmodules"])
    @skipIf(macos_version=["<", "10.12"])
    def test_expr(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_to_source_breakpoint(
            self, '// Set breakpoint here.', lldb.SBFileSpec('main.m'))

        self.runCmd(
            "settings set target.clang-module-search-paths \"" +
            self.getSourceDir() +
            "\"")

        self.expect("expr @import myModule; 3", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["int", "3"])

        self.expect("expr isInline(2)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["4"])
