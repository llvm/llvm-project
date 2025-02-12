import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftAsyncUnwind(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    @skipIf(oslist=["windows", "linux"])
    def test(self):
        """Test async unwinding with short backtraces work properly"""
        self.build()
        src = lldb.SBFileSpec("main.swift")
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "BREAK HERE", src
        )

        self.assertEqual(2, thread.GetNumFrames())
        self.assertIn("work", thread.GetFrameAtIndex(0).GetFunctionName())
        self.assertIn(
            "await resume partial function for implicit closure",
            thread.GetFrameAtIndex(1).GetFunctionName(),
        )
