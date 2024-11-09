import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftVariadicGenerics(TestBase):

    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()

        target,  process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec('main.swift'))
        self.expect('log enable lldb types')
        self.expect("frame variable v",
                    substrs=["a.Vector<4, Int>", "storage",
                             "0", "0",
                             "1", "1",
                             "2", "2",
                             "3", "3"])
