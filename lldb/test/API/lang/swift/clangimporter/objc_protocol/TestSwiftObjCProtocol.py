import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftObjCProtocol(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """Test printing an Objective-C protocol existential member."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift')
        )

        self.expect("v self", substrs=["obj", "0x"])
