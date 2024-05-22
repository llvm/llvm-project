import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestCase(lldbtest.TestBase):
    @swiftTest
    @skipUnlessFoundation
    def test(self):
        """Check that ClangImporter options can be overridden."""
        self.build()

        log = self.getBuildArtifact("lldb.log")
        self.runCmd(f"log enable lldb expr -f '{log}'")
        self.runCmd("settings set target.swift-clang-override-options x-DDELETEME=1")

        lldbutil.run_to_name_breakpoint(self, "main", bkpt_module="a.out")
        self.expect("expression 1")

        self.filecheck(f"platform shell cat {log}", __file__)
        # CHECK: CCC_OVERRIDE_OPTIONS: x-DDELETEME=1
        # CHECK: Deleting argument -DDELETEME=1
