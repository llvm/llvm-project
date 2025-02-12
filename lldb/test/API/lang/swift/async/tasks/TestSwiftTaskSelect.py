import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import TestBase
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):

    def test_backtrace_selected_task(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.runCmd("language swift task select task")
        self.expect(
            "thread backtrace",
            substrs=[
                ".sleep(",
                "`second() at main.swift:6:",
                "`first() at main.swift:2:",
                "`closure #1 in static Main.main() at main.swift:12:",
            ],
        )

    def test_navigate_selected_task_stack(self):
        self.build()
        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.runCmd("language swift task select task")

        thread = process.selected_thread
        self.assertEqual(thread.id, 2)
        self.assertEqual(thread.idx, 0xFFFFFFFF)
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
