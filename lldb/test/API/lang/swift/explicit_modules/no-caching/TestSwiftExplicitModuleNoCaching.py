import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftExplicitModuleNoCaching(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    def test(self):
        """
        Test that an uncached EBM build with a CAS config works.
        """
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec('main.swift')
        )
        log = self.getBuildArtifact("types.log")
        self.expect('log enable lldb types symbols -f "%s"' % log)
        self.expect("expression 1+1")
        self.filecheck_log(log, __file__)
        # CHECK: SwiftASTContextForExpressions(module: "a", cu: "main.swift")::ConfigureCASStorage() -- Setup CAS {{.*}}cas
        # CHECK: SwiftASTContextForExpressions(module: "a", cu: "main.swift")::LogConfiguration() -- Explicit module map entries
        # CHECK: SwiftASTContextForExpressions(module: "a", cu: "main.swift")::LogConfiguration() --     Swift
