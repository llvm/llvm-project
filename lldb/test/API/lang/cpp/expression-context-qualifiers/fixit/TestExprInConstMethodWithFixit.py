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
                "Possibly trying to mutate object in a const context. Try running the expression with",
                "expression --c++-ignore-context-qualifiers -- m_bar.method()",
            ],
        )

        # Two fix-its...
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

        # ...only emit a single hint.
        self.assertEqual(
            self.res.GetError().count(
                "Possibly trying to mutate object in a const context."
            ),
            1,
        )

        self.expect(
            "expression -Q -- m_bar->method()",
            error=True,
            substrs=["Evaluated this expression after applying Fix-It(s):"],
        )

        self.expect(
            "expression m_bar->method() + blah",
            error=True,
            substrs=[
                "member reference type 'const Bar' is not a pointer",
                "but function is not marked const",
                "Possibly trying to mutate object in a const context. Try running the expression with",
                "expression --c++-ignore-context-qualifiers -- m_bar.method() + blah",
            ],
        )

        self.expect(
            "expression -Q -- m_bar->method() + blah",
            error=True,
            substrs=[
                "Possibly trying to mutate object in a const context. Try running the expression with",
            ],
            matching=False,
        )
