import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftExplicitModules(lldbtest.TestBase):

    @skipEmbeddedSwift
    @swiftTest
    @skipIfWindows
    def test(self):
        """Test explicit Swift modules"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        log = self.getBuildArtifact("types.log")
        self.expect('log enable lldb types -f "%s"' % log)
        self.expect("expression c", substrs=['hello explicit'])
        self.filecheck_log(log, __file__)
        # CHECK: SwiftASTContextForExpressions(module: "a", cu: "main.swift"){{.*}} Discovered main module {{.*}}a.swiftmodule
        # CHECK: SwiftASTContextForExpressions(module: "a", cu: "main.swift"){{.*}} Module import remark: loaded module 'a'; source: '{{.*}}a.swiftmodule', loaded: '{{.*}}a.swiftmodule'

    @skipEmbeddedSwift
    @swiftTest
    @skipIfWindows
    def test_disable_esml(self):
        """Test disabling the explicit Swift module loader"""
        self.build()
        self.expect("settings set symbols.use-swift-explicit-module-loader false")
        self.expect("settings set target.experimental.swift-allow-implicit-module-loader true")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        log = self.getBuildArtifact("types.log")
        self.expect('log enable lldb types -f "%s"' % log)
        self.expect("expression c", substrs=["hello explicit"])
        self.filecheck_log(log, __file__, "--check-prefix=DISABLED")
        # DISABLED: SwiftASTContextForExpressions(module: "a", cu: "main.swift"){{.*}} Discovered main module{{.*}}a.swiftmodule
        # DISABLED: SwiftASTContextForExpressions(module: "a", cu: "main.swift"){{.*}} Module import remark: loaded module 'a'; source: 'a', loaded: 'a'
