"""
Test formatting of std::unordered_map related structures.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdUnorderedMapDataFormatterTestCase(TestBase):
    def check_ptr_or_ref(self, var_name: str):
        var = self.frame().FindVariable(var_name)
        self.assertTrue(var)

        pair = var.GetChildAtIndex(0)
        self.assertTrue(pair)

        self.assertEqual(pair.GetChildAtIndex(0).summary, '"Hello"')
        self.assertEqual(pair.GetChildAtIndex(1).summary, '"World"')

    def check_ptr_ptr(self, var_name: str):
        var = self.frame().FindVariable(var_name)
        self.assertTrue(var)

        ptr = var.GetChildAtIndex(0)
        self.assertTrue(ptr)

        pair = ptr.GetChildAtIndex(0)
        self.assertTrue(pair)

        self.assertEqual(pair.GetChildAtIndex(0).summary, '"Hello"')
        self.assertEqual(pair.GetChildAtIndex(1).summary, '"World"')

    def do_test(self):
        """Test that std::unordered_map related structures are formatted correctly when printed.
        Currently only tests format of std::unordered_map iterators.
        """
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
        self.check_ptr_or_ref("ref1")
        self.check_ptr_or_ref("ref2")
        self.check_ptr_or_ref("ref3")
        self.check_ptr_or_ref("ref4")
        self.check_ptr_or_ref("ref5")
        self.check_ptr_or_ref("ref6")

        # FIXME: we're getting this wrong.
        self.expect_var_path(
            "ref7",
            summary="size=0",
            type="const StringMapT *const &",
        )

    @add_test_categories(["libc++"])
    def test_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test()

    def do_test_ptr(self):
        """
        Test that pointers to std::unordered_map are formatted correctly.
        """

        (self.target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Stop here", lldb.SBFileSpec("main.cpp", False)
        )

        self.check_ptr_or_ref("ptr1")
        self.check_ptr_or_ref("ptr2")
        self.check_ptr_or_ref("ptr3")
        self.check_ptr_ptr("ptr4")
        self.check_ptr_ptr("ptr5")
        self.check_ptr_ptr("ptr6")

    @expectedFailureAll(
        bugnumber="https://github.com/llvm/llvm-project/issues/146040",
        compiler="clang",
        compiler_version=["<", "21"],
    )
    @add_test_categories(["libc++"])
    def test_ptr_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test_ptr()
