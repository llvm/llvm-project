"""
Test that we can call structors/destructors
annotated (and thus mangled) with ABI tags.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AbiTagStructorsTestCase(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec("main.cpp", False)
        )

        self.expect_expr("Tagged()", result_type="Tagged")
        self.expect_expr("t1 = t2", result_type="Tagged")

        self.expect("expr Tagged t3(t1)", error=False)
        self.expect("expr t1.~Tagged()", error=False)

        # Calls to deleting and base object destructor variants (D0 and D2 in Itanium ABI)
        self.expect_expr(
            "struct D : public HasVirtualDtor {}; D d; d.func()",
            result_type="int",
            result_value="10",
        )
