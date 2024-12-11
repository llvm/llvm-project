import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestSizedPointer(TestBase):

    mydir = TestBase.compute_mydir(__file__)


    def test(self):
        self.build()
        (_, self.process, _, bkpt) = lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.c"))

        self.expect_expr("strArg", result_type="char *")

        lldbutil.continue_to_breakpoint(self.process, bkpt)

        self.expect_expr("buf", result_type="int *")
        # Debugging will still work
        self.expect_expr("buf[len]", result_type="int")
        self.expect_expr("buf[len-1]", result_value='5')

        self.expect("expr str", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["Hello!"])
        self.expect_expr("ch", result_value="'!'")
