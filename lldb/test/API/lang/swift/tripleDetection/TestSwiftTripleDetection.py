import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftTripleDetection(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    def test(self):
        """Test that an underspecified triple is upgraded with a version number.
        """
        self.build()

        types_log = self.getBuildArtifact('types.log')
        self.runCmd('log enable lldb types -f "%s"' % types_log)
        exe = self.getBuildArtifact()
        arch = self.getArchitecture()
        target = self.dbg.CreateTargetWithFileAndTargetTriple(exe,
                                                              arch+"-apple-macos-unknown")
        bkpt = target.BreakpointCreateByName("main")
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.expect("expression 1")
        self.filecheck('platform shell cat "%s"' % types_log, __file__)
        # CHECK: {{SwiftASTContextForExpressions.*Preferring module triple .*-apple-macos.[0-9.]+ over target triple .*-apple-macos-unknown.}}
        # CHECK: {{SwiftASTContextForExpressions.*setting to ".*-apple-macos.[0-9.]+"}}
