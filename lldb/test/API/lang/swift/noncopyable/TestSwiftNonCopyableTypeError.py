import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftNonCopyableTypeError(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect("expression s", substrs=["Cannot evaluate an expression that results in a ~Copyable type"], error=True)

