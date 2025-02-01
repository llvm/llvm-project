import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftPrivateMember(TestBase):

    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()

        target,  process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec('main.swift'))
        frame = thread.frames[0]
        x = frame.FindVariable("x")
        member = x.GetChildAtIndex(0)
        self.assertIn("23", str(member))
