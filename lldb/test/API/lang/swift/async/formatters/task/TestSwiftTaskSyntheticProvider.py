import textwrap
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
            patterns=[
                textwrap.dedent(
                    r"""
                    \(Task<\(\), Error>\) task = id:(\d+) flags:(?:running\|)?(?:enqueued\|)?future \{
                      address = 0x[0-9a-f]+
                      id = \1
                      enqueuePriority = \.medium
                      children = \{\}
                    }
                    """
                ).strip()
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
            patterns=[
                textwrap.dedent(
                    r"""
                    \(UnsafeCurrentTask\) currentTask = id:(\d+) flags:(?:running\|)?(?:enqueued\|)?asyncLetTask|childTask|future \{
                      address = 0x[0-9a-f]+
                      id = \1
                      enqueuePriority = \.medium
                      children = \{\}
                    \}
                    """
                ).strip()
            ],
        )
