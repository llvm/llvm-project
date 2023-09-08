import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftArtificialSubclass(TestBase):
    @skipUnlessObjCInterop
    @swiftTest
    def test(self):
        """ Test that displaying an artificial type works correctly"""
        self.build()
        _, _, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        
        self.expect(
            "frame variable m",
            substrs=["Subclass)", "Superclass", "a = 42", "b = 97"]
        )
