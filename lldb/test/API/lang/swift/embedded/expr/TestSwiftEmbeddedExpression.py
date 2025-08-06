import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedExpression(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()

        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        types_log = self.getBuildArtifact('types.log')
        self.expect("log enable lldb types -v -f "+ types_log)

        self.expect("expr a.foo()", substrs=["(Int)", " = 16"])

        self.filecheck('platform shell cat "%s"' % types_log, __file__)
        # CHECK: [CheckFlagInCU] Found flag -enable-embedded-swift in CU:
