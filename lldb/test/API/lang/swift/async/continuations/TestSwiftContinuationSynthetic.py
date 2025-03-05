import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    @swiftTest
    def test_unsafe_continuation_printing(self):
        """Print an UnsafeContinuation and verify its children."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break unsafe continuation", lldb.SBFileSpec("main.swift")
        )
        self.expect(
            "v cont",
            substrs=[
                "(UnsafeContinuation<Void, Never>) cont = {",
                "task = {",
                "address = 0x",
                "id = ",
                "isFuture = true",
            ],
        )

    @swiftTest
    def test_checked_continuation_printing(self):
        """Print an CheckedContinuation and verify its children."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break checked continuation", lldb.SBFileSpec("main.swift")
        )
        self.expect(
            "v cont",
            substrs=[
                "(CheckedContinuation<Int, Never>) cont = {",
                "task = {",
                "address = 0x",
                "id = ",
                "isFuture = true",
            ],
        )
