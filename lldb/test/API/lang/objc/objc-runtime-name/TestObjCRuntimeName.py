"""
Test expressions involving a class whose Objective-C runtime name differs from
its source name, as is the case for Swift classes exposed to Objective-C.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestObjCRuntimeName(TestBase):
    @skipIf(compiler="clang", compiler_version=["<", "22.0"])
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.m"))

        # The runtime name only affects symbol names, the type keeps its
        # source name.
        self.expect_expr("self", result_type="Renamed *")

        # Expressions in a method are wrapped in a category on the class of
        # self, which references the class symbol.
        self.expect_expr("self.value", result_type="int", result_value="23")
        # Ivar access goes through the ivar offset symbol.
        self.expect_expr("self->_value", result_type="int", result_value="23")
        # Class message sends go through the class symbol.
        self.expect_expr("[Renamed classValue]", result_type="int", result_value="42")
