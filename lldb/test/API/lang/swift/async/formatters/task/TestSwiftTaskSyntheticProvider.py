import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    @swiftTest
    def test_top_level_task(self):
        """Test Task synthetic child provider for top-level Task."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break for top-level task", lldb.SBFileSpec("main.swift")
        )
        # Note: The value of isEnqueued is timing dependent. For that reason,
        # the test checks only that it has a value, not what the value is.
        self.expect(
            "frame var task",
            substrs=[
                "(Task<(), Error>) task = {",
                "isChildTask = false",
                "isFuture = true",
                "isGroupChildTask = false",
                "isAsyncLetTask = false",
                "isCancelled = false",
                "isEnqueued = ",
            ],
        )

    @swiftTest
    @skipIfLinux
    def test_current_task(self):
        """Test Task synthetic child for UnsafeCurrentTask (from an async let)."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break for current task", lldb.SBFileSpec("main.swift")
        )
        self.expect(
            "frame var currentTask",
            substrs=[
                "(UnsafeCurrentTask) currentTask = {",
                "isChildTask = true",
                "isFuture = true",
                "isGroupChildTask = false",
                "isAsyncLetTask = true",
                "isCancelled = false",
                "isEnqueued = false",
            ],
        )
