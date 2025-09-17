import textwrap
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    @swiftTest
    def test_print_task_group(self):
        """Print a TaskGroup and verify its children."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here TaskGroup", lldb.SBFileSpec("main.swift")
        )
        self.do_test_print()

    @swiftTest
    def test_print_throwing_task_group(self):
        """Print a ThrowingTaskGroup and verify its children."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here ThrowingTaskGroup", lldb.SBFileSpec("main.swift")
        )
        self.do_test_print()

    def do_test_print(self):
        self.expect(
            "v group",
            patterns=[
                textwrap.dedent(
                    r"""
                    \((?:Throwing)?TaskGroup<\(\)\??(?:, Error)?>\) group = \{
                      \[0\] = id:([1-9]\d*) flags:(?:suspended\|)?(?:running\|)?(?:enqueued\|)?groupChildTask \{
                        address = 0x[0-9a-f]+
                        id = \1
                        enqueuePriority = \.medium
                        parent = (.+)
                        children = \{\}
                      \}
                      \[1\] = id:([1-9]\d*) flags:(?:suspended\|)?(?:running\|)?(?:enqueued\|)?groupChildTask \{
                        address = 0x[0-9a-f]+
                        id = \3
                        enqueuePriority = \.medium
                        parent = \2
                        children = \{\}
                      \}
                      \[2\] = id:([1-9]\d*) flags:(?:suspended\|)?(?:running\|)?(?:enqueued\|)?groupChildTask \{
                        address = 0x[0-9a-f]+
                        id = \4
                        enqueuePriority = \.medium
                        parent = \2
                        children = \{\}
                      \}
                    \}
                    """
                ).strip()
            ],
        )

    @swiftTest
    def test_api_task_group(self):
        """Verify a TaskGroup contains its expected children."""
        self.build()
        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here TaskGroup", lldb.SBFileSpec("main.swift")
        )
        self.do_test_api(process)

    @swiftTest
    def test_api_throwing_task_group(self):
        """Verify a ThrowingTaskGroup contains its expected children."""
        self.build()
        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here ThrowingTaskGroup", lldb.SBFileSpec("main.swift")
        )
        self.do_test_api(process)

    def do_test_api(self, process):
        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()
        group = frame.FindVariable("group")
        self.assertEqual(group.num_children, 3)
        for task in group:
            self.assertEqual(str(task), str(group.GetChildMemberWithName(task.name)))
            self.assertEqual(
                task.GetChildMemberWithName("isGroupChildTask").summary, "true"
            )
