"""Show local variables and check that they can be inspected.

This test was added after we made a change in clang to normalize
DW_OP_constu(X < 32) to DW_OP_litX which broke the debugger because
it didn't read the value as an unsigned.
"""


from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LocalVariablesTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.source = "main.c"
        self.line = line_number(self.source, "// Set break point at this line.")

    def test_c_local_variables(self):
        """Test local variable value."""
        self.build()

        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(self.source), self.line)

        self.expect(
            "frame variable i",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["(unsigned int) i = 10"],
        )
