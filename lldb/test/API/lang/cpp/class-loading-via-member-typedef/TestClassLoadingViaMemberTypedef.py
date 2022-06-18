"""
Tests loading of classes when the loading is triggered via a typedef inside the
class (and not via the normal LLDB lookup that first resolves the surrounding
class).
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    @no_debug_info_test
    def test(self):
        self.build()
        self.createTestTarget()

        # Print the top-level typedef which triggers the loading of the class
        # that the typedef is defined inside.
        self.expect_expr(
            "pull_in_classes",
            result_type="StructWithMember::MemberTypedef",
            result_value="0",
        )

        # Print the classes and check their types.
        self.expect_expr(
            "struct_to_print",
            result_type="StructWithMember",
            result_children=[
                ValueCheck(
                    name="m",
                    type="StructWithNested::Nested<int>::OtherTypedef",
                    children=[ValueCheck(name="i", value="0", type="int")],
                )
            ],
        )
