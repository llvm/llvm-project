import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbarm64e import Arm64eTestBase


class TestPtrauthBlendInterp(Arm64eTestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    def test_interpreter_reads_ptrauth_field(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c", False)
        )

        # Disable the JIT, so that IRInterpreter evaluates the expression.
        options = lldb.SBExpressionOptions()
        options.SetAllowJIT(False)
        self.expect_expr(
            "*s_ptr->p", result_type="int", result_value="123", options=options
        )
