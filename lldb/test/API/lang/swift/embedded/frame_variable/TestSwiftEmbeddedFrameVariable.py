import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftEmbeddedFrameVariable(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()

        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect(
            "frame variable varB",
            substrs=["varB = ", "a = (field = 4.2000000000000002)", "b = 123456"],
        )
        self.expect(
            "frame variable tuple",
            substrs=[
                "(a.A, a.B) tuple = {",
                "0 = (field = 4.2000000000000002)",
                "1 = {",
                "a = (field = 4.2000000000000002)",
                "b = 123456",
            ],
        )

        # TODO: test enums when "rdar://119343683 (Embedded Swift trivial case enum fails to link)" is solved
