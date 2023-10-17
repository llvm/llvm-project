from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftLateDylib(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @swiftTest
    @skipIfDarwinEmbedded
    def test(self):
        """Test that a late loaded Swift dylib is debuggable"""
        arch = self.getArchitecture()
        self.build(dictionary={"TRIPLE": arch + "-apple-macosx11.0.0", "ARCH": arch,
                               "DYLIB_TRIPLE": arch + "-apple-macosx12.0.0"})
        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec("main.swift"))
        self.expect("expr -- import Dylib")
        # Scan through the types log.
        self.filecheck('platform shell cat "%s"' % log, __file__)
# CHECK: SwiftASTContextForExpressions::LogConfiguration(){{.*}}Architecture{{.*}}-apple-macosx11.0.0
# CHECK-NOT: __PATH_FROM_DYLIB__
#       Verify that the deployment target didn't change:
# CHECK: SwiftASTContextForExpressions::LogConfiguration(){{.*}}Architecture{{.*}}-apple-macosx11.0.0
#       But LLDB has picked up extra paths:
# CHECK: SwiftASTContextForExpressions::LogConfiguration(){{.*}}__PATH_FROM_DYLIB__
