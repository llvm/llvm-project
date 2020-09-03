import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestNamespaceLocalVarSameNameCppAndC(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @add_test_categories(["gmodules"])
    def test_namespace_local_var_same_name_cpp_and_c(self):
        self.build()

        (self.target, self.process, _, bkpt) = lldbutil.run_to_source_breakpoint(self, '// break here',
                lldb.SBFileSpec("main.cpp", False))

        self.expect_expr("error", result_type="int", result_value="1")
        lldbutil.continue_to_breakpoint(self.process, bkpt)
        self.expect_expr("error", result_type="int", result_value="1")
