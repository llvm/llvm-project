import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftAsyncExpressions(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    @skipIfWindows
    @skipIfLinux
    @skipIf(archs=no_match(["arm64", "arm64e", "arm64_32", "x86_64"]))
    def test_actor(self):
        """Test async unwind"""
        self.build()
        target, process, thread, main_bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec("main.swift"))
        self.expect("expr n", substrs=["42"])
        process.Continue()
        stop_desc = process.GetSelectedThread().GetStopDescription(1024)
        self.assertNotIn("EXC_BAD_ACCESS", stop_desc)
