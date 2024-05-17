import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftFrameworkPaths(lldbtest.TestBase):

    @swiftTest
    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipIf(oslist=no_match(["macosx"]))
    def test_system_framework(self):
        """Test the discovery of framework search paths from framework dependencies."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        log = self.getBuildArtifact("types.log")
        self.expect('log enable lldb types -f "%s"' % log)
        self.expect("expression -- 0")
        self.filecheck('platform shell cat "%s"' % log, __file__,
                       '--check-prefix=CHECK_SYS')
        # CHECK_SYS: SwiftASTContextForExpressions::LogConfiguration(){{.*}}/secret_path

    @swiftTest
    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipIf(oslist=no_match(["macosx"]))
    def test_module_context_framework_path(self):
        """Test the discovery of framework search paths from framework dependencies."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        self.expect('settings set symbols.swift-validate-typesystem false')
        self.expect('settings set symbols.use-swift-typeref-typesystem false')
        log = self.getBuildArtifact("types.log")
        self.expect('log enable lldb types -f "%s"' % log)
        self.expect("expression -- d", substrs=["member"])
        self.filecheck('platform shell cat "%s"' % log, __file__,
                       '--check-prefix=CHECK_MOD')
        # CHECK_MOD: SwiftASTContextForModule("a.out")::LogConfiguration(){{.*}}other_secret_path
