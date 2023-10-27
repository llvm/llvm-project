import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprBug35310(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def test_issue35310(self):
        """Test invoking functions with non-standard linkage names.

        The GNU abi_tag extension used by libstdc++ is a common source
        of these, but they could originate from other reasons as well.
        """
        self.build()

        lldbutil.run_to_source_breakpoint(self, "// Break here", self.main_source_spec)

        self.expect_expr("a.test_abi_tag()", result_value="1")
        self.expect_expr("a.test_asm_name()", result_value="2")
