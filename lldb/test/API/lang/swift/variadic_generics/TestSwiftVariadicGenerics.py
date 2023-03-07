import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2

class TestSwiftVariadicGenerics(TestBase):

    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()

        target,  _, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec('a.swift'))

        self.expect("frame variable",
                    substrs=["Pack{(a.A, a.B)}", "args", "i = 23", "d = 2.71"])

