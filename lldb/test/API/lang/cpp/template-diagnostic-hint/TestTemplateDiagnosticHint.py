import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @expectedFailureWindows
    def test(self):
        self.build()
        (_, process, _, _) = lldbutil.run_to_source_breakpoint(
            self, "main", lldb.SBFileSpec("main.cpp")
        )

        self.expect(
            "expression some_template_func<int, long>(5)",
            error=True,
            substrs=[
                "does not name a template but is followed by template arguments",
                "note: Naming template instantiation not yet supported.",
                "Template functions can be invoked via their mangled name.",
            ],
        )

        self.expect(
            "expression some_template_func<int, long>(5) + some_template_func<int, long>(5)",
            error=True,
            substrs=[
                "does not name a template but is followed by template arguments",
                "does not name a template but is followed by template arguments",
            ],
        )

        self.assertEqual(
            self.res.GetError().count(
                "note: Naming template instantiation not yet supported"
            ),
            1,
        )

        self.expect(
            "expression Foo<int>::smethod()",
            error=True,
            substrs=[
                "no template named 'Foo'",
                "note: Naming template instantiation not yet supported.",
                "Template functions can be invoked via their mangled name.",
            ],
        )
