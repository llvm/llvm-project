import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftClangImporterCaching(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipIf(setting=('symbols.swift-precise-compiler-invocation', 'false'))
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """
        Test flipping on/off implicit modules.
        """
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.swift'))
        log = self.getBuildArtifact("types.log")
        self.runCmd("settings set target.swift-clang-override-options +-DADDED=1")
        self.runCmd("settings set target.swift-extra-clang-flags -- -DEXTRA=1")
        self.expect('log enable lldb types -f "%s"' % log)
        self.expect("expression obj", DATA_TYPES_DISPLAYED_CORRECTLY,
                    substrs=["b ="])
        self.filecheck('platform shell cat "%s"' % log, __file__)
### -cc1 should be round-tripped so there is no more `-cc1` in the extra args. Look for `-triple` which is a cc1 flag.
#       CHECK:  SwiftASTContextForExpressions(module: "a", cu: "main.swift")::LogConfiguration() --     -triple
### Check include paths in the module are forwards. The first argument is the source directory.
#       CHECK:  SwiftASTContextForExpressions(module: "a", cu: "main.swift")::LogConfiguration() --     -I
#       CHECK-NEXT:  SwiftASTContextForExpressions(module: "a", cu: "main.swift")::LogConfiguration() --
#       CHECK:  SwiftASTContextForExpressions(module: "a", cu: "main.swift")::LogConfiguration() --     -I
#       CHECK-NEXT:  SwiftASTContextForExpressions(module: "a", cu: "main.swift")::LogConfiguration() --     /TEST_DIR
#       CHECK:  SwiftASTContextForExpressions(module: "a", cu: "main.swift")::LogConfiguration() --     -F
#       CHECK-NEXT:  SwiftASTContextForExpressions(module: "a", cu: "main.swift")::LogConfiguration() --     /FRAMEWORK_DIR
#       CHECK:  SwiftASTContextForExpressions(module: "a", cu: "main.swift")::LogConfiguration() --     -DADDED=1
#       CHECK:  SwiftASTContextForExpressions(module: "a", cu: "main.swift")::LogConfiguration() --     -DEXTRA=1
#       CHECK:  SwiftASTContextForExpressions(module: "a", cu: "main.swift") Module import remark: loaded module 'ClangA'
#       CHECK-NOT: -cc1
#       CHECK-NOT: -fmodule-file-cache-key
#       CHECK-NOT: Clang error:
