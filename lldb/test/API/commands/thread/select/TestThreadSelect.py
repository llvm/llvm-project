import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *


class TestCase(TestBase):
    def test_invalid_arg(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect(
            "thread select 0x1ffffffff",
            error=True,
            startstr="error: Invalid thread index '0x1ffffffff'",
        )
        self.expect(
            "thread select -t 0x1ffffffff",
            error=True,
            startstr="error: Invalid thread ID",
        )
        self.expect(
            "thread select 1 2 3",
            error=True,
            startstr="error: 'thread select' takes exactly one thread index argument, or a thread ID option:",
        )
        self.expect(
            "thread select -t 1234 1",
            error=True,
            startstr="error: 'thread select' cannot take both a thread ID option and a thread index argument:",
        )
        # Parses but not a valid thread id.
        self.expect(
            "thread select 0xffffffff",
            error=True,
            startstr="error: Invalid thread index #0xffffffff.",
        )
        self.expect(
            "thread select -t 0xffffffff",
            error=True,
            startstr="error: Invalid thread ID",
        )

    def test_thread_select_tid(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        self.runCmd(
            "thread select -t %d" % self.thread().GetThreadID(),
        )
