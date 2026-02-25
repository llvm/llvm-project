import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftExplicitModuleCaching(TestBase):

    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipIf(setting=('symbols.swift-precise-compiler-invocation', 'false'))
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec('main.swift')
        )
        log = self.getBuildArtifact("types.log")
        self.expect('log enable lldb types symbols -f "%s"' % log)
        self.expect("expression obj", DATA_TYPES_DISPLAYED_CORRECTLY,
                    substrs=["b ="])

        frame = thread.frames[0]
        data = frame.GetLanguageSpecificData()
        has_cas = data.GetValueForKey("SwiftHasCAS").GetBooleanValue()
        self.assertTrue(has_cas)

        ## Check temporarily disabled until the swift feature 'debug-info-explicit-dependency' is enabled
        # self.filecheck_log(log, __file__)
# CHECK: SwiftASTContextForExpressions(module: "a", cu: "main.swift")::DiscoverExplicitMainModule() -- Discovered main module llvmcas://
