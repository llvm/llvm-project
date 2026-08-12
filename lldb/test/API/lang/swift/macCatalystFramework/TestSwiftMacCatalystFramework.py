import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftMacCatalystFramework(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @skipIf(macos_version=["<", "26"])
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    def test(self):
        """A macCatalyst (arm64-apple-ios-macabi) executable that loads a plain
           macOS (arm64-apple-macosx) Swift framework. When stopped in the
           framework's Swift code, LLDB must build the framework's
           SwiftASTContextForExpressions with the framework's *macOS* triple,
           not the executable's macCatalyst triple.
        """
        self.build()
        types_log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % types_log)

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("Framework.swift"),
            extra_images=["Framework.framework/Framework"])

        arch = self.getArchitecture()
        self.expect("image list -t -b",
                    patterns=[arch + r".*-apple-ios.*-macabi a\.out"])

        self.expect("expression -- x", substrs=['"hello"'])
        self.filecheck_log(types_log, __file__)
#       CHECK: SwiftASTContextForExpressions(module: "Framework", cu: "Framework.swift")::LogConfiguration(){{.*}}Architecture{{.*}}-apple-macosx
