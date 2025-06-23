import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftAsyncBreakpoints(lldbtest.TestBase):
    @swiftTest
    @skipIfLinux
    def test(self):
        """Test async that async breakpoints are not filtered when the same
        statement is present across multiple funclets"""
        self.build()
        filespec = lldb.SBFileSpec("main.swift")
        target, process, thread, breakpoint1 = lldbutil.run_to_source_breakpoint(
            self, "breakpoint_start", filespec
        )
        breakpoint = target.BreakpointCreateBySourceRegex("breakhere", filespec)
        self.assertEquals(breakpoint.GetNumLocations(), 2)

        process.Continue()
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)
        self.assertEquals(thread.GetStopDescription(128), "breakpoint 2.1")
        self.expect("expr argument", substrs=["1"])

        process.Continue()
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)
        self.assertEquals(thread.GetStopDescription(128), "breakpoint 2.2")
        self.expect("expr argument", substrs=["2"])
