import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftVariadicGenerics(TestBase):

    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()

        target,  process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec('main.swift'))
        self.expect("frame variable ints",
                    substrs=["a.Vector<4, Int>", "storage",
                             "0", "1", "2", "3"])
        self.expect("frame variable bools",
                    substrs=["a.Vector<2, Bool>", "storage",
                             "false", "true"])
        self.expect("frame variable structs",
                    substrs=["a.Vector<2, S>", "storage",
                             "i", "1", "j", "2", "i", "3", "j", "4"])
