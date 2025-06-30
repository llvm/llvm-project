"""
Test formatting of std::unordered_map related structures.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxUnorderedMapDataFormatterTestCase(TestBase):
    def check_reference(self, var_name: str, expected_type: str):
        self.expect_var_path(
            var_name,
            summary="size=1",
            type=expected_type,
            children=[
                ValueCheck(
                    name="[0]",
                    children=[
                        ValueCheck(name="first", summary='"Hello"'),
                        ValueCheck(name="second", summary='"World"'),
                    ],
                ),
            ],
        )

    @add_test_categories(["libc++"])
    def test_iterator_formatters(self):
        """Test that std::unordered_map related structures are formatted correctly when printed.
        Currently only tests format of std::unordered_map iterators.
        """
        self.build()
        (self.target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec("main.cpp", False)
        )

        # Test empty iterators
        self.expect_expr("empty_iter", "")
        self.expect_expr("const_empty_iter", "")

        lldbutil.continue_to_breakpoint(process, bkpt)

        # Check that key/value is correctly formatted
        self.expect_expr(
            "foo",
            result_children=[
                ValueCheck(name="first", summary='"Foo"'),
                ValueCheck(name="second", summary='"Bar"'),
            ],
        )

        # Check invalid iterator is empty
        self.expect_expr("invalid", "")

        # Const key/val iterator
        self.expect_expr(
            "const_baz",
            result_children=[
                ValueCheck(name="first", summary='"Baz"'),
                ValueCheck(name="second", summary='"Qux"'),
            ],
        )

        # Bucket iterators
        # I.e., std::__hash_map_const_iterator<const_local_iterator<...>>
        # and std::__hash_map_iterator<local_iterator<...>>
        self.expect_expr(
            "bucket_it",
            result_children=[
                ValueCheck(name="first", summary='"Baz"'),
                ValueCheck(name="second", summary='"Qux"'),
            ],
        )

        self.expect_expr(
            "const_bucket_it",
            result_children=[
                ValueCheck(name="first", summary='"Baz"'),
                ValueCheck(name="second", summary='"Qux"'),
            ],
        )

        lldbutil.continue_to_breakpoint(process, bkpt)

        # Test references to std::unordered_map
        self.check_reference("ref1", "const StringMapT &")
        self.check_reference("ref2", "StringMapT &")
        self.check_reference("ref3", "StringMapTRef")
        self.check_reference("ref4", "const StringMapT &")
        self.check_reference("ref5", "const StringMapT &&")
        self.check_reference("ref6", "StringMapT &&")

        # FIXME: we're getting this wrong.
        self.expect_var_path(
            "ref7",
            summary="size=0",
            type="const StringMapT *const &",
        )
