import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftResilienceSuperclassOtherMod(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('ModWithClass.swift'))

        self.expect("expression c.v", substrs=["Int", "42"])
