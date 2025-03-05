import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    @swiftTest
    def test_value_printing(self):
        """Print a TaskGroup and verify its children."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect(
            "v group",
            substrs=[
                "[0] = {",
                "address = 0x",
                "id = ",
                "isGroupChildTask = true",
                "[1] = {",
                "address = 0x",
                "id = ",
                "isGroupChildTask = true",
                "[2] = {",
                "address = 0x",
                "id = ",
                "isGroupChildTask = true",
            ],
        )

    @swiftTest
    def test_value_api(self):
        """Verify a TaskGroup contains its expected children."""
        self.build()
        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()
        group = frame.FindVariable("group")
        self.assertEqual(group.num_children, 3)
        for task in group:
            self.assertEqual(str(task), str(group.GetChildMemberWithName(task.name)))
            self.assertEqual(
                task.GetChildMemberWithName("isGroupChildTask").summary, "true"
            )
