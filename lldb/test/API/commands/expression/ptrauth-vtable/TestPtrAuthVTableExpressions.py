"""
VTable pointers are signed with a discriminator that incorporates the object's
address (PointerAuthVTPtrAddressDiscrimination) and class type (
PointerAuthVTPtrTypeDiscrimination).
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestPtrAuthVTableExpressions(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessArm64eSupported
    def test_virtual_call_on_debuggee_object(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        self.expect_expr("d.value()", result_type="int", result_value="20")
        self.expect_expr("od.value()", result_type="int", result_value="30")

    @skipUnlessArm64eSupported
    def test_virtual_call_through_base_pointer(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        self.expect_expr("base_ptr->value()", result_type="int", result_value="20")

    @skipUnlessArm64eSupported
    def test_virtual_call_via_helper(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        self.expect_expr("call_value(&d)", result_type="int", result_value="20")
        self.expect_expr("call_value(&od)", result_type="int", result_value="30")
        self.expect_expr("call_value(base_ptr)", result_type="int", result_value="20")
