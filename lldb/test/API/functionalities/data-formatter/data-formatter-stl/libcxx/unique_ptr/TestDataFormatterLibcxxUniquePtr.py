"""
Test lldb data formatter for libc++ std::unique_ptr.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    @add_test_categories(["libc++"])
    def test_unique_ptr_variables(self):
        """Test `frame variable` output for `std::unique_ptr` types."""
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        valobj = self.expect_var_path(
            "up_empty",
            type="std::unique_ptr<int, std::default_delete<int> >",
            summary="nullptr",
            children=[ValueCheck(name="__value_")],
        )
        self.assertEqual(
            valobj.child[0].GetValueAsUnsigned(lldb.LLDB_INVALID_ADDRESS), 0
        )

        self.expect(
            "frame variable *up_empty", substrs=["(int) *up_empty = <parent is NULL>"]
        )

        valobj = self.expect_var_path(
            "up_int",
            type="std::unique_ptr<int, std::default_delete<int> >",
            summary="10",
            children=[ValueCheck(name="__value_")],
        )
        self.assertNotEqual(valobj.child[0].unsigned, 0)

        valobj = self.expect_var_path(
            "up_int_ref",
            type="std::unique_ptr<int, std::default_delete<int> > &",
            summary="10",
            children=[ValueCheck(name="__value_")],
        )
        self.assertNotEqual(valobj.child[0].unsigned, 0)

        valobj = self.expect_var_path(
            "up_int_ref_ref",
            type="std::unique_ptr<int, std::default_delete<int> > &&",
            summary="10",
            children=[ValueCheck(name="__value_")],
        )
        self.assertNotEqual(valobj.child[0].unsigned, 0)

        valobj = self.expect_var_path(
            "up_str",
            type="std::unique_ptr<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::default_delete<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >",
            summary='"hello"',
            children=[ValueCheck(name="__value_", summary='"hello"')],
        )

        valobj = self.expect_var_path(
            "up_user", type="std::unique_ptr<User, std::default_delete<User> >"
        )
        self.assertRegex(valobj.summary, "^User @ 0x0*[1-9a-f][0-9a-f]+$")
        self.assertNotEqual(valobj.child[0].unsigned, 0)

        valobj = self.expect_var_path(
            "*up_user",
            type="User",
            children=[
                ValueCheck(name="id", value="30"),
                ValueCheck(name="name", summary='"steph"'),
            ],
        )
        self.assertEqual(str(valobj), '(User) *__value_ = (id = 30, name = "steph")')

        self.expect_var_path("up_user->id", type="int", value="30")
        self.expect_var_path("up_user->name", type="std::string", summary='"steph"')
