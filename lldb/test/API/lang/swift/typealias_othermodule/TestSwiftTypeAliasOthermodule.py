from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftTypeAliasOtherModule(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    def test(self):
        """Test that type aliases can be imported from reflection metadata"""
        arch = self.getArchitecture()
        self.build()
        self.expect('settings set symbols.swift-load-conformances true')
        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb expr types -f "%s"' % log)
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"), extra_images=["Dylib"]
        )
        self.expect("frame variable -- payload", substrs=["Bool", "true"])
        self.expect("expr -- payload", substrs=["Dylib.Impl.Payload", "true"])
        self.expect("continue")
        self.expect("frame variable -- payload", substrs=["Bool", "true"])
        self.expect(
            "expr -- payload", substrs=["Dylib.GenericImpl<Bool>.Payload", "true"]
        )
