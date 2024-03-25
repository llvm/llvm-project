"""
Test LLDB's std::ranges::ref_view formatter
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxRangesRefViewDataFormatterTestCase(TestBase):
    def check_string_vec_children(self):
        return [
            ValueCheck(name="[0]", summary='"First"'),
            ValueCheck(name="[1]", summary='"Second"'),
            ValueCheck(name="[2]", summary='"Third"'),
            ValueCheck(name="[3]", summary='"Fourth"'),
        ]

    def check_string_vec_ref_view(self):
        return ValueCheck(
            name="*__range_",
            summary="size=4",
            children=self.check_string_vec_children(),
        )

    def check_foo(self):
        return ValueCheck(name="vec", children=self.check_string_vec_children())

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler="clang", compiler_version=["<", "16.0"])
    def test_with_run_command(self):
        """Test that std::ranges::ref_view is formatted correctly when printed."""
        self.build()
        (self.target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec("main.cpp", False)
        )

        # Check ref_view over a std::string
        self.expect_var_path(
            "single", children=[ValueCheck(name="*__range_", summary='"First"')]
        )

        # Check all_view, which is a ref_view in this case
        self.expect_var_path("all", children=[self.check_string_vec_ref_view()])

        # Check take_view format. Embeds a ref_view
        self.expect_var_path(
            "subset",
            children=[
                ValueCheck(children=[self.check_string_vec_ref_view()]),
                ValueCheck(name="__count_", value="2"),
            ],
        )

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        # Check ref_view over custom type 'struct Foo'
        self.expect_var_path(
            "view",
            children=[
                ValueCheck(
                    name="*__range_",
                    children=[
                        ValueCheck(name="[0]", type="Foo", children=[self.check_foo()]),
                        ValueCheck(name="[1]", type="Foo", children=[self.check_foo()]),
                    ],
                )
            ],
        )
