import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftAsyncBreakpoints(lldbtest.TestBase):
    @swiftTest
    @skipIfWindows
    @skipIfLinux
    @skipIf(archs=no_match(["arm64", "arm64e", "x86_64"]))
    def test(self):
        """Test async breakpoints"""
        self.build()
        filespec = lldb.SBFileSpec("main.swift")
        target, process, thread, breakpoint1 = lldbutil.run_to_source_breakpoint(
            self, "Breakpoint1", filespec
        )
        breakpoint2 = target.BreakpointCreateBySourceRegex("Breakpoint2", filespec)
        breakpoint3 = target.BreakpointCreateBySourceRegex("Breakpoint3", filespec)
        self.assertEquals(breakpoint1.GetNumLocations(), 2)
        self.assertEquals(breakpoint2.GetNumLocations(), 1)
        self.assertEquals(breakpoint3.GetNumLocations(), 2)

        location11 = breakpoint1.GetLocationAtIndex(0)
        location12 = breakpoint1.GetLocationAtIndex(1)
        self.assertEquals(location11.GetHitCount(), 1)
        self.assertEquals(location12.GetHitCount(), 0)

        self.assertEquals(thread.GetStopDescription(128), "breakpoint 1.1")
        process.Continue()
        self.assertEquals(thread.GetStopDescription(128), "breakpoint 1.2")

        thread.StepOver()
        self.assertEquals(thread.GetStopDescription(128), "breakpoint 2.1")
        self.expect("expr timestamp1", substrs=["42"])

        thread.StepOver()
        self.assertIn("breakpoint 3.1", thread.GetStopDescription(128))
        self.expect("expr timestamp1", substrs=["42"])

        process.Continue()
        self.assertIn("breakpoint 3.2", thread.GetStopDescription(128))
        self.expect("expr timestamp1", substrs=["42"])

        thread.StepOver()
        self.expect("expr timestamp1", substrs=["42"])
        self.expect("expr timestamp2", substrs=["43"])

        self.runCmd("settings set language.enable-filter-for-line-breakpoints false")
        breakpoint1_no_filter = target.BreakpointCreateBySourceRegex(
            "Breakpoint1", filespec
        )
        self.assertEquals(breakpoint1_no_filter.GetNumLocations(), 3)
