import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        (_, process, _, _) = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect(
            "expression m_bar->method()",
            error=True,
            substrs=[
                "member reference type 'const Bar' is not a pointer",
                "but function is not marked const",
            ],
        )

        # Two fix-its
        self.expect(
            "expression -- m_bar->method() + m_bar->method()",
            error=True,
            substrs=[
                "member reference type 'const Bar' is not a pointer",
                "but function is not marked const",
                "member reference type 'const Bar' is not a pointer",
                "but function is not marked const",
            ],
        )

        self.expect(
            "expression m_bar->method() + blah",
            error=True,
            substrs=[
                "member reference type 'const Bar' is not a pointer",
                "but function is not marked const",
            ],
        )
