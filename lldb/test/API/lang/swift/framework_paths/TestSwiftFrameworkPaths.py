import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


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
        # CHECK_SYS: SwiftASTContextForExpressions(module: "a", cu: "main.swift")::LogConfiguration(){{.*}}/secret_path
