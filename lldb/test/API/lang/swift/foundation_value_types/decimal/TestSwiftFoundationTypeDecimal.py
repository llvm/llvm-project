import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @skipEmbeddedSwift
    @swiftTest
    @skipIf(oslist=["linux", "windows"])
    def test_decimal(self):
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.GetSelectedFrame()
        expected = {
            "x": "42.500000",
            "y": "422.500000",
            "z": "-23.600000",
            "patatino": "12345567888.123394",
        }
        for name, summary in expected.items():
            lldbutil.check_variable(self, frame.FindVariable(name), summary=summary)
            lldbutil.check_variable(
                self, frame.EvaluateExpression(name), summary=summary
            )
