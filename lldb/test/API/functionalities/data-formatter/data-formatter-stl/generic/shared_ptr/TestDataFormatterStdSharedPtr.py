"""
Test lldb data formatter for std::shared_ptr.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def do_test(self):
        """Test `frame variable` output for `std::shared_ptr` types."""
        (_, process, _, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        valobj = self.expect_var_path(
            "sp_empty",
            type="std::shared_ptr<int>",
            summary="nullptr",
            children=[ValueCheck(name="pointer")],
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
            children=[ValueCheck(name="pointer")],
        )
        self.assertRegex(valobj.summary, r"^10( strong=1)? weak=0$")
        self.assertNotEqual(valobj.child[0].unsigned, 0)

        valobj = self.expect_var_path(
            "sp_int_ref",
            type="std::shared_ptr<int> &",
            children=[ValueCheck(name="pointer")],
        )
        self.assertRegex(valobj.summary, r"^10( strong=1)? weak=0$")
        self.assertNotEqual(valobj.child[0].unsigned, 0)

        valobj = self.expect_var_path(
            "sp_int_ref_ref",
            type="std::shared_ptr<int> &&",
            children=[ValueCheck(name="pointer")],
        )
        self.assertRegex(valobj.summary, r"^10( strong=1)? weak=0$")
        self.assertNotEqual(valobj.child[0].unsigned, 0)

        valobj = self.expect_var_path(
            "sp_str",
            children=[ValueCheck(name="pointer", summary='"hello"')],
        )
        self.assertRegex(valobj.summary, r'^"hello"( strong=1)? weak=0$')

        valobj = self.expect_var_path("sp_user", type="std::shared_ptr<User>")
        self.assertRegex(
            valobj.summary,
            "element_type @ 0x0*[1-9a-f][0-9a-f]+( strong=1)? weak=0",
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
        self.assertEqual(str(valobj), '(User) *pointer = (id = 30, name = "steph")')

        self.expect_var_path("sp_user->id", type="int", value="30")
        self.expect_var_path("sp_user->name", type="std::string", summary='"steph"')

        valobj = self.expect_var_path(
            "si", type="std::shared_ptr<int>", summary="47 strong=2 weak=0"
        )

        valobj = self.expect_var_path(
            "sie", type="std::shared_ptr<int>", summary="nullptr strong=2 weak=0"
        )

        lldbutil.continue_to_breakpoint(process, bkpt)

        valobj = self.expect_var_path(
            "si", type="std::shared_ptr<int>", summary="47 strong=2 weak=2"
        )
        valobj = self.expect_var_path(
            "sie", type="std::shared_ptr<int>", summary="nullptr strong=2 weak=2"
        )
        valobj = self.expect_var_path(
            "wie", type="std::weak_ptr<int>", summary="nullptr strong=2 weak=2"
        )

        self.expect_var_path("si.pointer", type="int *")
        self.expect_var_path("*si.pointer", type="int", value="47")
        self.expect_var_path("si.object", type="int", value="47")

        self.runCmd("settings set target.experimental.use-DIL true")
        self.expect_var_path("ptr_node->value", value="1")
        self.expect_var_path("ptr_node->next->value", value="2")
        self.expect_var_path("(*ptr_node).value", value="1")
        self.expect_var_path("(*(*ptr_node).next).value", value="2")

    @add_test_categories(["libc++"])
    def test_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test()

    @add_test_categories(["libstdcxx"])
    def test_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test()

    @add_test_categories(["msvcstl"])
    def test_msvcstl(self):
        # No flags, because the "msvcstl" category checks that the MSVC STL is used by default.
        self.build()
        self.do_test()
