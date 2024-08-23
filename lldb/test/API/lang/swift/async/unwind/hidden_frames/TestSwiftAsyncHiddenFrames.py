import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftAsyncHiddenFrames(lldbtest.TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @skipIf(oslist=['windows', 'linux'])
    def test(self):
        """Test async unwind"""
        self.build()
        src = lldb.SBFileSpec('main.swift')
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', src)

        self.expect("thread backtrace", ordered=True, patterns=[
            "frame.*closure.*main",
            "frame.*closure.*C.run",
            "frame.*Main.main"
        ])
        self.expect("thread backtrace", matching=False, patterns=[
            "frame.*partial apply",
            "frame.*back deployment fallback"
        ])
