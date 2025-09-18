import textwrap
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect(
            "language swift task info",
            patterns=[
                textwrap.dedent(
                    r"""
                    \(UnsafeCurrentTask\) current_task = id:1 flags:(?:running|enqueued) \{
                      address = (0x[0-9a-f]+)
                      id = 1
                      enqueuePriority = 0
                      parent = nil
                      children = \{
                        0 = id:2 flags:(?:suspended\|)?(?:running\|)?(?:enqueued\|)?asyncLetTask \{
                          address = 0x[0-9a-f]+
                          id = 2
                          enqueuePriority = \.medium
                          parent = \1 \{\}
                          children = \{\}
                        \}
                      \}
                    \}
                    """
                ).strip()
            ],
        )
