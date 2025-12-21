import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftExplicitModules(lldbtest.TestBase):

    @swiftTest
    def test(self):
        """Test explicit Swift modules"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        log = self.getBuildArtifact("types.log")
        self.expect('log enable lldb types -f "%s"' % log)
        self.expect("expression c", substrs=['hello explicit'])
        self.filecheck('platform shell cat "%s"' % log, __file__)
        # CHECK: SwiftASTContextForExpressions(module: "a", cu: "main.swift"){{.*}} found explicit module {{.*}}a.swiftmodule
        # CHECK: SwiftASTContextForExpressions(module: "a", cu: "main.swift"){{.*}} Module import remark: loaded module 'a'; source: '{{.*}}a.swiftmodule', loaded: '{{.*}}a.swiftmodule'

    @swiftTest
    def test_disable_esml(self):
        """Test disabling the explicit Swift module loader"""
        self.build()
        self.expect("settings set symbols.use-swift-explicit-module-loader false")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        log = self.getBuildArtifact("types.log")
        self.expect('log enable lldb types -f "%s"' % log)
        self.expect("expression c", substrs=['hello explicit'])
        self.filecheck('platform shell cat "%s"' % log, __file__, '--check-prefix=DISABLED')
        # DISABLED: SwiftASTContextForExpressions(module: "a", cu: "main.swift"){{.*}} found explicit module {{.*}}a.swiftmodule
        # DISABLED: SwiftASTContextForExpressions(module: "a", cu: "main.swift"){{.*}} Module import remark: loaded module 'a'; source: 'a', loaded: 'a'

        
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
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect('expression URL(string: "https://lldb.llvm.org")',
                    error=True)
        self.expect("expression import Foundation")
        self.expect('expression URL(string: "https://lldb.llvm.org")',
                    substrs=["https://lldb.llvm.org"])
