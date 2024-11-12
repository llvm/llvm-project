import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os


class TestSwiftOtherArchDylib(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @skipIf(archs=no_match(["arm64"]))
    @skipIf(archs=["arm64"], bugnumber="the swift.org toolchain cannot produce arm64e binaries")
    def test(self):
        """Test module import from dylibs with an architecture
           that uses a different SDK"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_name_breakpoint(
            self, "foo", extra_images=['OtherArch.framework/OtherArch'])
        types_log = self.getBuildArtifact('types.log')
        self.expect("log enable lldb types -f "+ types_log)

        self.expect("expression 1", substrs=['1'])
        frame = thread.frames[0]
        while 'OtherArch.foo' not in frame.name:
            frame = frame.parent
            self.assertIsNotNone(frame)
        thread.SetSelectedFrame(frame.idx)
        self.expect("expression 1", substrs=['1'])

        # Check the types log.
        self.filecheck('platform shell cat "%s"' % types_log, __file__)
        # CHECK: SwiftASTContextForExpressions::LogConfiguration() arm64-apple-macosx
        # CHECK: Enabling per-module Swift scratch context
        # CHECK: {{SwiftASTContextForExpressions..OtherArch..}}::LogConfiguration() arm64e-apple-macosx
