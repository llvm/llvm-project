"""
Tests GCC's complex integer types.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.c"))

        # FIXME: LLDB treats all complex ints as unsigned, so the value is wrong.
        self.expect_expr("complex_int", result_type="_Complex int", result_value="4294967295 + 4294967294i")
        self.expect_expr("complex_long", result_type="_Complex long")

        self.expect_expr("complex_unsigned", result_type="_Complex int", result_value="1 + 2i")
        self.expect_expr("complex_unsigned_long", result_type="_Complex long", result_value="1 + 2i")

    @no_debug_info_test
    def test_long_long(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.c"))

        # FIXME: We get the type wrong if long has the same size as long long.
        # FIXME: LLDB treats all complex ints as unsigned, so the value is wrong.
        self.expect_expr("complex_long_long", result_type="_Complex long")
        self.expect_expr("complex_unsigned_long_long", result_type="_Complex long", result_value="1 + 2i")
