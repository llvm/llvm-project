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
            substrs=[
                "(UnsafeCurrentTask) current_task = {",
                "id = 1",
                "isChildTask = false",
                "isAsyncLetTask = false",
                "children = {",
                "0 = {",
                "id = 2",
                "isChildTask = true",
                "isAsyncLetTask = true",
                "children = {}",
            ],
        )
