"""
Test lldb data formatter for std::unique_ptr.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def do_test(self):
        """Test `frame variable` output for `std::unique_ptr` types."""

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        valobj = self.expect_var_path(
            "up_empty",
            summary="nullptr",
            children=[ValueCheck(name="pointer")],
        )
        self.assertEqual(
            valobj.child[0].GetValueAsUnsigned(lldb.LLDB_INVALID_ADDRESS), 0
        )

        self.expect(
            "frame variable *up_empty", substrs=["(int) *up_empty = <parent is NULL>"]
        )

        valobj = self.expect_var_path(
            "up_int",
            summary="10",
            children=[ValueCheck(name="pointer")],
        )
        self.assertNotEqual(valobj.child[0].unsigned, 0)

        valobj = self.expect_var_path(
            "up_int_ref",
            summary="10",
            children=[ValueCheck(name="pointer")],
        )
        self.assertNotEqual(valobj.child[0].unsigned, 0)

        valobj = self.expect_var_path(
            "up_int_ref_ref",
            summary="10",
            children=[ValueCheck(name="pointer")],
        )
        self.assertNotEqual(valobj.child[0].unsigned, 0)

        valobj = self.expect_var_path(
            "up_str",
            summary='"hello"',
            children=[ValueCheck(name="pointer", summary='"hello"')],
        )

        valobj = self.expect_var_path("up_user")
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
        self.assertEqual(str(valobj), '(User) *pointer = (id = 30, name = "steph")')

        valobj = self.expect_var_path(
            "up_non_empty_deleter",
            type="std::unique_ptr<int, NonEmptyIntDeleter>",
            summary="1234",
            children=[
                ValueCheck(name="pointer"),
                ValueCheck(
                    name="deleter", children=[ValueCheck(name="dummy_", value="9999")]
                ),
            ],
        )
        self.assertNotEqual(valobj.child[0].unsigned, 0)

        self.expect_var_path("up_user->id", type="int", value="30")
        self.expect_var_path("up_user->name", type="std::string", summary='"steph"')

        self.runCmd("settings set target.experimental.use-DIL true")
        self.expect_var_path("ptr_node->value", value="1")
        self.expect_var_path("ptr_node->next->value", value="2")
        self.expect_var_path("(*ptr_node).value", value="1")
        self.expect_var_path("(*(*ptr_node).next).value", value="2")

    @add_test_categories(["libstdcxx"])
    def test_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test()

    @add_test_categories(["libc++"])
    def test_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test()

    @add_test_categories(["msvcstl"])
    def test_msvcstl(self):
        # No flags, because the "msvcstl" category checks that the MSVC STL is used by default.
        self.build()
        self.do_test()

    def do_test_recursive_unique_ptr(self):
        # Tests that LLDB can handle when we have a loop in the unique_ptr
        # reference chain and that it correctly handles the different options
        # for the frame variable command in this case.
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_source_regexp(self, "Set break point at this line.")
        self.runCmd("run", RUN_SUCCEEDED)
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        self.expect("frame variable f1->next", substrs=["next = NodeU @"])
        self.expect(
            "frame variable --ptr-depth=1 f1->next",
            substrs=["next = NodeU @", "value = 2"],
        )
        self.expect(
            "frame variable --ptr-depth=2 f1->next",
            substrs=["next = NodeU @", "value = 1", "value = 2"],
        )

        frame = self.frame()
        self.assertTrue(frame.IsValid())
        self.assertEqual(
            2,
            frame.GetValueForVariablePath("f1->next.object.value").GetValueAsUnsigned(),
        )
        self.assertEqual(
            2, frame.GetValueForVariablePath("f1->next->value").GetValueAsUnsigned()
        )
        self.assertEqual(
            1,
            frame.GetValueForVariablePath(
                "f1->next.object.next.obj.value"
            ).GetValueAsUnsigned(),
        )
        self.assertEqual(
            1,
            frame.GetValueForVariablePath("f1->next->next->value").GetValueAsUnsigned(),
        )

    @add_test_categories(["libstdcxx"])
    def test_recursive_unique_ptr_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test_recursive_unique_ptr()

    @add_test_categories(["libc++"])
    def test_recursive_unique_ptr_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test_recursive_unique_ptr()

    @add_test_categories(["msvcstl"])
    def test_recursive_unique_ptr_msvcstl(self):
        self.build()
        self.do_test_recursive_unique_ptr()
