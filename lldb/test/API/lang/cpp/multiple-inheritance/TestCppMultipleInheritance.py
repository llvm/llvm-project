import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        # Member access
        self.expect_expr("C.Base1::m_base", result_type="int", result_value="11")
        self.expect_expr("C.Base2::m_base", result_type="int", result_value="12")
        self.expect_expr("C.m1", result_type="int", result_value="22")
        self.expect_expr("C.m2", result_type="int", result_value="33")
        self.expect_expr("C.m_final", result_type="int", result_value="44")

        # Virtual functions
        self.expect_expr("C.Base1::virt_base()", result_type="int", result_value="111")
        self.expect_expr("C.Base2::virt_base()", result_type="int", result_value="121")
        self.expect_expr("C.virt1()", result_type="int", result_value="3")
        self.expect_expr("C.virt2()", result_type="int", result_value="5")
        self.expect_expr("C.final_virt()", result_type="int", result_value="7")
        self.expect_expr("C.virt_common()", result_type="int", result_value="444")

        # Normal functions
        self.expect_expr("C.Base1::func_base()", result_type="int", result_value="112")
        self.expect_expr("C.Base2::func_base()", result_type="int", result_value="122")
        self.expect_expr("C.func1()", result_type="int", result_value="4")
        self.expect_expr("C.func2()", result_type="int", result_value="6")
        self.expect_expr("C.final_func()", result_type="int", result_value="8")
        self.expect_expr("C.func_common()", result_type="int", result_value="888")
