import shutil
import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestCase(lldbtest.TestBase):
    @swiftTest
    def test_missing_explicit_modules(self):
        """Test missing explicit Swift modules and fallback to implicit modules."""
        self.build()

        # This test verifies the case where explicit modules are missing.
        # Remove explicit modules from their place in the module cache.
        mod_cache = self.getBuildArtifact("private-module-cache")
        shutil.rmtree(mod_cache)

        lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec("main.swift")
        )

        log = self.getBuildArtifact("types.log")
        self.runCmd(f"log enable lldb types -f '{log}'")

        self.expect("expression c", substrs=["hello implicit fallback"])

        self.filecheck(f"platform shell cat {log}", __file__)
        # CHECK: Nonexistent explicit module file
        # CHECK: Explicit modules : false

    @swiftTest
    def test_sanity(self):
        """Check the normal behavior."""
        self.build()

        # This test verifies the case where explicit modules are missing.
        # Remove explicit modules from their place in the module cache.
        mod_cache = self.getBuildArtifact("private-module-cache")

        lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec("main.swift")
        )

        log = self.getBuildArtifact("types.log")
        self.runCmd(f"log enable lldb types -f '{log}'")

        self.expect("expression c")
        self.expect("expression -- import Foundation")

        self.filecheck(f"platform shell cat {log}", __file__,
                       '--check-prefix=CHECK-SANITY')
        # CHECK-SANITY: Explicit modules : true
        # CHECK-SANITY: Turning off implicit modules
        # CHECK-SANITY: Turning on implicit modules
