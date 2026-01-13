"""
Tests that we can call functions that have definitions in multiple
CUs in the debug-info (which is the case for functions defined in headers).
The linker will most likely de-duplicate the functiond definitions when linking
the final executable. On Darwin, this will create a debug-map that LLDB will use
to fix up object file addresses to addresses in the linked executable. However,
if we parsed the DIE from the object file whose functiond definition got stripped
by the linker, LLDB needs to ensure it can still resolve the function symbol it
got for it.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestFunctionCallFromObjectFile(TestBase):
    def test_lib1(self):
        self.build()
        lldbutil.run_to_name_breakpoint(self, "lib1_func")

        self.expect_expr("Foo{}.foo()", result_type="int", result_value="15")

    def test_lib2(self):
        self.build()
        lldbutil.run_to_name_breakpoint(self, "lib2_func")

        self.expect_expr("Foo{}.foo()", result_type="int", result_value="15")
