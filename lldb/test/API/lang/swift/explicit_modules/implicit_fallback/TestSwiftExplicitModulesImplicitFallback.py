import shutil
import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestCase(lldbtest.TestBase):
    @swiftTest
    @skipIf(oslist=["linux"], bugnumber="rdar://124691219")
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
