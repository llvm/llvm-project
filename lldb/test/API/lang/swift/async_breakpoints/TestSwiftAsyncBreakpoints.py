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
        breakpoint4 = target.BreakpointCreateBySourceRegex("Breakpoint4", filespec)
        breakpoint5 = target.BreakpointCreateBySourceRegex("Breakpoint5", filespec)
        self.assertEquals(breakpoint1.GetNumLocations(), 1)
        self.assertEquals(breakpoint2.GetNumLocations(), 1)
        self.assertEquals(breakpoint3.GetNumLocations(), 1)
        # FIXME: there should be two breakpoints here, but the "entry" funclet of the
        # implicit closure is mangled slightly differently. rdar://147035260
        self.assertEquals(breakpoint4.GetNumLocations(), 3)
        self.assertEquals(breakpoint5.GetNumLocations(), 1)

        location11 = breakpoint1.GetLocationAtIndex(0)
        self.assertEquals(location11.GetHitCount(), 1)

        self.assertEquals(thread.GetStopDescription(128), "breakpoint 1.1")
        process.Continue()

        self.assertEquals(thread.GetStopDescription(128), "breakpoint 2.1")
        self.expect("expr timestamp1", substrs=["42"])

        thread.StepOver()
        self.assertIn("breakpoint 3.1", thread.GetStopDescription(128))
        self.expect("expr timestamp1", substrs=["42"])

        self.runCmd("settings set language.enable-filter-for-line-breakpoints false")
        breakpoint1_no_filter = target.BreakpointCreateBySourceRegex(
            "Breakpoint1", filespec
        )
        self.assertEquals(breakpoint1_no_filter.GetNumLocations(), 2)
