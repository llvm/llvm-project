import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import TestBase
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect(
            "language swift task backtrace task",
            substrs=[
                ".sleep(",
                "`second() at main.swift:6",
                "`first() at main.swift:2",
                "`closure #1 in static Main.main() at main.swift:12",
            ],
        )
