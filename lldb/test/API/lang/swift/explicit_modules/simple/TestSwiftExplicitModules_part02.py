import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftExplicitModules(lldbtest.TestBase):

    @skipEmbeddedSwift
    @swiftTest
    @skipUnlessDarwin
    def test_import(self):
        """Test an implicit import inside an explicit build"""
        mod_cache = self.getBuildArtifact("my-clang-modules-cache")
        if os.path.isdir(mod_cache):
          shutil.rmtree(mod_cache)

        self.runCmd('settings set symbols.clang-modules-cache-path "%s"'
                    % mod_cache)

        self.build()
        self.expect('log enable lldb types')
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect('expression URL(string: "https://lldb.llvm.org")',
                    error=True)
        self.expect("expression import Foundation")
        self.expect('expression URL(string: "https://lldb.llvm.org")',
                    substrs=["https://lldb.llvm.org"])
