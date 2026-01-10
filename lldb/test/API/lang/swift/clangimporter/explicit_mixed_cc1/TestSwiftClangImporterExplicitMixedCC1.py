import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftClangImporterExplicitCC1(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """
        Test detection of mixing and matching of options in a catch-all "*" SwiftASTContext.
        """
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('c.c'))
        log = self.getBuildArtifact("types.log")
        self.expect('log enable lldb types -f "%s"' % log)
        self.expect("expr -l Swift -- 1+1")
        self.filecheck('platform shell cat "%s"' % log, __file__)
#       CHECK: SwiftASTContextForExpressions(module: "{{.*}}", cu: "*")::AddDiagnostic() -- Mixing and matching of cc1 and driver options detected
