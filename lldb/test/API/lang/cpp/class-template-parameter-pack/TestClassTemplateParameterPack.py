import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(compiler="gcc")
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.cpp"))

        # Test non-type template parameter packs.
        self.expect_expr("myC", result_type="C<int, 16, 32>", result_children=[
            ValueCheck(name="C<int, 16>", children=[
                ValueCheck(name="member", value="64")
            ])
        ])
        self.expect_expr("myLesserC.argsAre_16_32()", result_value="false")
        self.expect_expr("myC.argsAre_16_32()", result_value="true")

        # Test type template parameter packs.
        self.expect_expr("myD", result_type="D<int, int, bool>", result_children=[
            ValueCheck(name="D<int, int>", children=[
                ValueCheck(name="member", value="64")
            ])
        ])
        self.expect_expr("myLesserD.argsAre_Int_bool()", result_value="false")
        self.expect_expr("myD.argsAre_Int_bool()", result_value="true")

        # Disabling until we do template lookup correctly: http://lists.llvm.org/pipermail/lldb-commits/Week-of-Mon-20180507/040689.html
        # FIXME: Rewrite this with expect_expr
        # self.expect("expression -- C<int, 16>().isSixteenThirtyTwo()", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["false"])
        # self.expect("expression -- C<int, 16, 32>().isSixteenThirtyTwo()", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["true"])
        # self.expect("expression -- D<int, int>().isIntBool()", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["false"])
        # self.expect("expression -- D<int, int, bool>().isIntBool()", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["true"])
