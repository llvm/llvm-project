import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import TestBase
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):

    @swiftTest
    def test_backtrace_selected_task_variable(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.do_backtrace_selected_task("task")

    @swiftTest
    def test_backtrace_selected_task_address(self):
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.frames[0]
        task = frame.FindVariable("task")
        task_addr = task.GetChildMemberWithName("address").unsigned
        self.do_backtrace_selected_task(task_addr)

    def do_backtrace_selected_task(self, arg):
        self.runCmd(f"language swift task select {arg}")
        self.expect(
            "thread backtrace",
            substrs=[
                ".sleep(",
                "`second() at main.swift:6:",
                "`first() at main.swift:2:",
                "`closure #1() at main.swift:12:",
            ],
        )

    @swiftTest
    def test_navigate_stack_of_selected_task_variable(self):
        self.build()
        _, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.do_test_navigate_selected_task_stack(process, "task")

    @swiftTest
    def test_navigate_stack_of_selected_task_address(self):
        self.build()
        _, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.frames[0]
        task = frame.FindVariable("task")
        task_addr = task.GetChildMemberWithName("address").unsigned
        self.do_test_navigate_selected_task_stack(process, task_addr)

    def do_test_navigate_selected_task_stack(self, process, arg):
        self.runCmd(f"language swift task select {arg}")
        thread = process.GetSelectedThread()

        self.assertIn(
            "libswift_Concurrency.", thread.GetSelectedFrame().module.file.basename
        )

        frame_idx = -1
        for frame in thread:
            if "`second()" in str(frame):
                frame_idx = frame.idx
        self.assertNotEqual(frame_idx, -1)

        self.expect(f"frame select {frame_idx}", substrs=[f"frame #{frame_idx}:"])
        frame = thread.GetSelectedFrame()
        self.assertIn(".second()", frame.function.name)

        self.expect("up", substrs=[f"frame #{frame_idx + 1}:"])
        frame = thread.GetSelectedFrame()
        self.assertIn(".first()", frame.function.name)

        self.expect("up", substrs=[f"frame #{frame_idx + 2}:"])
        frame = thread.GetSelectedFrame()
        self.assertIn(".Main.main()", frame.function.name)
