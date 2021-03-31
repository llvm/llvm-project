"""
Test lldb data formatter for libc++ std::shared_ptr.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    def test_shared_ptr_variables(self):
        """Test `frame variable` output for `std::shared_ptr` types."""
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        valobj = self.expect_var_path(
            "sp_empty",
            type="std::shared_ptr<int>",
            summary="nullptr",
            children=[ValueCheck(name="__ptr_")],
        )
        self.assertEqual(
            valobj.child[0].GetValueAsUnsigned(lldb.LLDB_INVALID_ADDRESS), 0
        )

        self.expect(
            "frame variable *sp_empty", substrs=["(int) *sp_empty = <parent is NULL>"]
        )

        valobj = self.expect_var_path(
            "sp_int",
            type="std::shared_ptr<int>",
            children=[ValueCheck(name="__ptr_")],
        )
        self.assertRegex(valobj.summary, r"^10( strong=1)? weak=1$")
        self.assertNotEqual(valobj.child[0].unsigned, 0)

        valobj = self.expect_var_path(
            "sp_int_ref",
            type="std::shared_ptr<int> &",
            children=[ValueCheck(name="__ptr_")],
        )
        self.assertRegex(valobj.summary, r"^10( strong=1)? weak=1$")
        self.assertNotEqual(valobj.child[0].unsigned, 0)

        valobj = self.expect_var_path(
            "sp_int_ref_ref",
            type="std::shared_ptr<int> &&",
            children=[ValueCheck(name="__ptr_")],
        )
        self.assertRegex(valobj.summary, r"^10( strong=1)? weak=1$")
        self.assertNotEqual(valobj.child[0].unsigned, 0)

        valobj = self.expect_var_path(
            "sp_str",
            type="std::shared_ptr<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >",
            children=[ValueCheck(name="__ptr_", summary='"hello"')],
        )
        self.assertRegex(valobj.summary, r'^"hello"( strong=1)? weak=1$')

        valobj = self.expect_var_path("sp_user", type="std::shared_ptr<User>")
        self.assertRegex(
            valobj.summary,
            "^std(::__1)?::shared_ptr<User>::element_type @ 0x0*[1-9a-f][0-9a-f]+( strong=1)? weak=1",
        )
        self.assertNotEqual(valobj.child[0].unsigned, 0)

        valobj = self.expect_var_path(
            "*sp_user",
            type="User",
            children=[
                ValueCheck(name="id", value="30"),
                ValueCheck(name="name", summary='"steph"'),
            ],
        )
        self.assertEqual(str(valobj), '(User) *__ptr_ = (id = 30, name = "steph")')

        self.expect_var_path("sp_user->id", type="int", value="30")
        self.expect_var_path("sp_user->name", type="std::string", summary='"steph"')
