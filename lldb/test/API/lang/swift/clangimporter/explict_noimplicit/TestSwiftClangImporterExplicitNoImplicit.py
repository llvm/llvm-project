import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftClangImporterExplicitNoImplicit(TestBase):

    NO_DEBUG_INFO_TESTCASE = True
    
    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """
        Test flipping on/off implicit modules.
        """
        self.build()
        self.expect('settings set symbols.clang-modules-cache-path '
                    + self.getBuildArtifact("IMPLICIT-CLANG-MODULE-CACHE"))
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.swift'),
                                          extra_images=['Dylib'])
        log = self.getBuildArtifact("types.log")
        self.expect('log enable lldb types -f "%s"' % log)
        self.expect('settings set target.experimental.swift-allow-implicit-module-loader false')
        # This is expected to fail because the dependencies of Dylib
        # (including the stdlib) are implicit modules.
        self.expect("expression obj", error=True)
        self.expect('settings set target.experimental.swift-allow-implicit-module-loader true')
        self.expect("expression obj", DATA_TYPES_DISPLAYED_CORRECTLY,
                    substrs=["hidden"])
        self.filecheck('platform shell cat "%s"' % log, __file__)
        # CHECK-NOT: IMPLICIT-CLANG-MODULE-CACHE/{{.*}}/SwiftShims-{{.*}}.pcm
        # CHECK: IMPLICIT-CLANG-MODULE-CACHE/{{.*}}/Hidden-{{.*}}.pcm
