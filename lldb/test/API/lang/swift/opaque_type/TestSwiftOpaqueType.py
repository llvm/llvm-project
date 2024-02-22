import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftOpaque(TestBase):

    @swiftTest
    def test(self):
        """Test opaque types in parameter positions"""         
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec("main.swift"))
        self.expect("v p", substrs=["C", "23"])

