import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    @swiftTest
    def test_value_printing(self):
        """Print an UnsafeContinuation and verify its children."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect(
            "v cont",
            substrs=[
                "(UnsafeContinuation<Void, Never>) cont = {",
                "task = {",
                "isFuture = true",
            ],
        )
