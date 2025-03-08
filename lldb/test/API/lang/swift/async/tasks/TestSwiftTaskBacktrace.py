import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import TestBase
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):

    def test_backtrace_task_variable(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.do_backtrace("task")

    def test_backtrace_task_address(self):
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.frames[0]
        task = frame.FindVariable("task")
        task_addr = task.GetChildMemberWithName("address").unsigned
        self.do_backtrace(task_addr)

    def do_backtrace(self, arg):
        self.expect(
            f"language swift task backtrace {arg}",
            substrs=[
                ".sleep(",
                "`second() at main.swift:6",
                "`first() at main.swift:2",
                "`closure #1 in static Main.main() at main.swift:12",
            ],
        )
