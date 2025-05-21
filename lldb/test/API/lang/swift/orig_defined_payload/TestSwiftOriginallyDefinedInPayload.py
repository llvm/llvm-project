import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os


class TestSwiftOriginallyDefinedInPayload(TestBase):
    @swiftTest
    def test(self):
        self.build()

        # rdar://151579199
        self.runCmd("setting set symbols.swift-validate-typesystem false")

        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect(
            "frame variable c",
            substrs=[
                "c = 1 value",
                "flexible = {",
                "0 = (value = 100)",
                "1 = (value = 200)",
                " 2 = (value = 300)",
            ],
        )
