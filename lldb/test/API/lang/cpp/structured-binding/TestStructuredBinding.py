import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestStructuredBinding(TestBase):

    @skipIf(oslist=["linux"], archs=["arm"])
    @skipIf(compiler="clang", compiler_version=['<', '14.0'])
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.cpp"))

        self.expect_expr("a1", result_type="A",
            result_children=[ValueCheck(name="x", type="int"),
                             ValueCheck(name="y", type="int")])
        self.expect_expr("b1", result_type="char", result_value="'a'")
        self.expect_expr("c1", result_type="char", result_value="'b'")
        self.expect_expr("d1", result_type="short", result_value="50")
        self.expect_expr("e1", result_type="int", result_value="60")
        self.expect_expr("f1", result_type="char", result_value="'c'")

        self.expect_expr("a2", result_type="A",
            result_children=[ValueCheck(name="x", type="int"),
                             ValueCheck(name="y", type="int")])
        self.expect_expr("b2", result_type="char", result_value="'a'")
        self.expect_expr("c2", result_type="char", result_value="'b'")
        self.expect_expr("d2", result_type="short", result_value="50")
        self.expect_expr("e2", result_type="int", result_value="60")
        self.expect_expr("f2", result_type="char", result_value="'c'")

        self.expect_expr("a3", result_type="A",
            result_children=[ValueCheck(name="x", type="int"),
                             ValueCheck(name="y", type="int")])
        self.expect_expr("b3", result_type="char", result_value="'a'")
        self.expect_expr("c3", result_type="char", result_value="'b'")
        self.expect_expr("d3", result_type="short", result_value="50")
        self.expect_expr("e3", result_type="int", result_value="60")
        self.expect_expr("f3", result_type="char", result_value="'c'")

        self.expect_expr("carr_ref1", result_type="char", result_value="'a'")
        self.expect_expr("carr_ref2", result_type="char", result_value="'b'")
        self.expect_expr("carr_ref3", result_type="char", result_value="'c'")

        self.expect_expr("sarr_ref1", result_type="short", result_value="11")
        self.expect_expr("sarr_ref2", result_type="short", result_value="12")
        self.expect_expr("sarr_ref3", result_type="short", result_value="13")

        self.expect_expr("iarr_ref1", result_type="int", result_value="22")
        self.expect_expr("iarr_ref2", result_type="int", result_value="33")
        self.expect_expr("iarr_ref3", result_type="int", result_value="44")

        self.expect_expr("carr_rref1", result_type="char", result_value="'a'")
        self.expect_expr("carr_rref2", result_type="char", result_value="'b'")
        self.expect_expr("carr_rref3", result_type="char", result_value="'c'")

        self.expect_expr("sarr_rref1", result_type="short", result_value="11")
        self.expect_expr("sarr_rref2", result_type="short", result_value="12")
        self.expect_expr("sarr_rref3", result_type="short", result_value="13")

        self.expect_expr("iarr_rref1", result_type="int", result_value="22")
        self.expect_expr("iarr_rref2", result_type="int", result_value="33")
        self.expect_expr("iarr_rref3", result_type="int", result_value="44")

        self.expect_expr("carr_copy1", result_type="char", result_value="'a'")
        self.expect_expr("carr_copy2", result_type="char", result_value="'b'")
        self.expect_expr("carr_copy3", result_type="char", result_value="'c'")

        self.expect_expr("sarr_copy1", result_type="short", result_value="11")
        self.expect_expr("sarr_copy2", result_type="short", result_value="12")
        self.expect_expr("sarr_copy3", result_type="short", result_value="13")

        self.expect_expr("iarr_copy1", result_type="int", result_value="22")
        self.expect_expr("iarr_copy2", result_type="int", result_value="33")
        self.expect_expr("iarr_copy3", result_type="int", result_value="44")

        self.expect_expr("tx1", result_value="4")
        self.expect_expr("ty1", result_value="'z'")
        self.expect_expr("tz1", result_value="10")

        self.expect_expr("tx2", result_value="4")
        self.expect_expr("ty2", result_value="'z'")
        self.expect_expr("tz2", result_value="10")
