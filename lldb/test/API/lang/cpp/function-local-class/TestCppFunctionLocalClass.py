import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        m_val = self.expect_expr(
            "m",
            result_type="WithMember",
            result_children=[ValueCheck(name="i", value="1")],
        )
        # FIXME: The non-display name doesn't include the function, so users
        # can't actually match specific classes by their name. Either document
        # or fix this.
        self.assertEqual(m_val.GetType().GetName(), "WithMember")
        # Try accessing the type in the expression evaluator.
        self.expect_expr("m.i", result_type="int", result_value="1")

        self.expect_expr(
            "typedef_unnamed",
            result_type="TypedefUnnamed",
            result_children=[ValueCheck(name="a", value="2")],
        )
        self.expect_expr(
            "typedef_unnamed2",
            result_type="TypedefUnnamed2",
            result_children=[ValueCheck(name="b", value="3")],
        )
        self.expect_expr(
            "unnamed",
            result_type="(unnamed struct)",
            result_children=[ValueCheck(name="i", value="4")],
        )
        self.expect_expr(
            "unnamed2",
            result_type="(unnamed struct)",
            result_children=[ValueCheck(name="j", value="5")],
        )

        # Try a class that is only forward declared.
        self.expect_expr("fwd", result_type="Forward *")
        self.expect(
            "expression -- fwd->i",
            error=True,
            substrs=["member access into incomplete type 'Forward'"],
        )
        self.expect(
            "expression -- *fwd",
            error=True,
            substrs=["incomplete type 'Forward' where a complete type is required"],
        )

        # Try a class that has a name that matches a class in the global scope.
        self.expect_expr("fwd_conflict", result_type="ForwardConflict *")
        # FIXME: This pulls in the unrelated type with the same name from the
        # global scope.
        self.expect(
            "expression -- fwd_conflict->i",
            error=True,
            substrs=[
                # This should point out that ForwardConflict is incomplete.
                "no member named 'i' in 'ForwardConflict'"
            ],
        )
        self.expect(
            "expression -- *fwd_conflict",
            error=True,
            substrs=[
                # This should fail to parse instead.
                "couldn't read its memory"
            ],
        )
