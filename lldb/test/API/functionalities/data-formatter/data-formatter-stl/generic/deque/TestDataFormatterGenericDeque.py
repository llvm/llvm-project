import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

USE_LIBSTDCPP = "USE_LIBSTDCPP"
USE_LIBCPP = "USE_LIBCPP"

class GenericDequeDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def findVariable(self, name):
        var = self.frame().FindVariable(name)
        self.assertTrue(var.IsValid())
        return var

    def getVariableType(self, name):
        var = self.findVariable(name)
        return var.GetType().GetDisplayTypeName()

    def check_size(self, var_name, size):
        var = self.findVariable(var_name)
        self.assertEqual(var.GetNumChildren(), size)


    def do_test(self, stdlib_type):
        self.build(dictionary={stdlib_type: '1'})
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec("main.cpp"))

        self.expect_expr("empty", result_children=[])
        self.expect_expr("deque_1", result_children=[
            ValueCheck(name="[0]", value="1"),
        ])
        self.expect_expr("deque_3", result_children=[
            ValueCheck(name="[0]", value="3"),
            ValueCheck(name="[1]", value="1"),
            ValueCheck(name="[2]", value="2")
        ])

        self.check_size("deque_200_small", 200)
        for i in range(0, 100):
            self.expect_var_path("deque_200_small[%d]"%(i), children=[
                ValueCheck(name="a", value=str(-99 + i)),
                ValueCheck(name="b", value=str(-100 + i)),
                ValueCheck(name="c", value=str(-101 + i)),
            ])
            self.expect_var_path("deque_200_small[%d]"%(i + 100), children=[
                ValueCheck(name="a", value=str(i)),
                ValueCheck(name="b", value=str(1 + i)),
                ValueCheck(name="c", value=str(2 + i)),
            ])

        self.check_size("deque_200_large", 200)
        for i in range(0, 100):
            self.expect_var_path("deque_200_large[%d]"%(i), children=[
                ValueCheck(name="a", value=str(-99 + i)),
                ValueCheck(name="b", value=str(-100 + i)),
                ValueCheck(name="c", value=str(-101 + i)),
                ValueCheck(name="d")
            ])
            self.expect_var_path("deque_200_large[%d]"%(i + 100), children=[
                ValueCheck(name="a", value=str(i)),
                ValueCheck(name="b", value=str(1 + i)),
                ValueCheck(name="c", value=str(2 + i)),
                ValueCheck(name="d")
            ])

    @add_test_categories(["libstdcxx"])
    def test_libstdcpp(self):
        self.do_test(USE_LIBSTDCPP)

    @add_test_categories(["libc++"])
    def test_libcpp(self):
         self.do_test(USE_LIBCPP)
