import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftExprMissingType(lldbtest.TestBase):
    @swiftTest
    def test(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect("frame variable invisible", substrs=["1", "2"])
        self.expect("expr invisible", error=True,
                    substrs=["Missing debug info", "invisible"])
