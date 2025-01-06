"""
Test that we can set up software breakpoint even if we failed to decode and execute instruction
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestBreakpointIllegal(TestBase):
    @skipIf(archs=no_match(["rv64gc"]))
    def test_4(self):
        self.build()
        (target, process, cur_thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "main", lldb.SBFileSpec("main.c")
        )
        self.runCmd("thread step-inst")
        # we need to step more, as some compilers do not set appropriate debug info.
        while cur_thread.GetStopDescription(256) == "instruction step into":
            self.runCmd("thread step-inst")
        # The stop reason of the thread should be illegal opcode.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_SIGNAL,
            substrs=["stopped", "stop reason = signal SIGILL: illegal opcode"],
        )

    @skipIf(archs=no_match(["rv64gc"]))
    def test_2(self):
        self.build(dictionary={"C_SOURCES": "compressed.c", "EXE": "compressed.x"})
        (target, process, cur_thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "main", lldb.SBFileSpec("compressed.c"), exe_name="compressed.x"
        )
        self.runCmd("thread step-inst")
        # we need to step more, as some compilers do not set appropriate debug info.
        while cur_thread.GetStopDescription(256) == "instruction step into":
            self.runCmd("thread step-inst")
        # The stop reason of the thread should be illegal opcode.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_SIGNAL,
            substrs=["stopped", "stop reason = signal SIGILL: illegal opcode"],
        )
