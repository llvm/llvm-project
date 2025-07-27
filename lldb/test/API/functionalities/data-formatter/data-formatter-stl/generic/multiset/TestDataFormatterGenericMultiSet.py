"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class GenericMultiSetDataFormatterTestCase(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.namespace = "std"

    def findVariable(self, name):
        var = self.frame().FindVariable(name)
        self.assertTrue(var.IsValid())
        return var

    def getVariableType(self, name):
        var = self.findVariable(name)
        return var.GetType().GetDisplayTypeName()

    def check(self, var_name, size):
        var = self.findVariable(var_name)
        self.assertEqual(var.GetNumChildren(), size)
        children = []
        for i in range(size):
            child = var.GetChildAtIndex(i)
            children.append(ValueCheck(value=child.GetValue()))
        self.expect_var_path(
            var_name, type=self.getVariableType(var_name), children=children
        )

    def do_test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        (self.target, process, _, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line.", lldb.SBFileSpec("main.cpp", False)
        )

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd("type format clear", check=False)
            self.runCmd("type summary clear", check=False)
            self.runCmd("type filter clear", check=False)
            self.runCmd("type synth clear", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        ii_type = self.getVariableType("ii")
        self.assertTrue(
            ii_type.startswith(self.namespace + "::multiset"), "Type: " + ii_type
        )

        self.expect("frame variable ii", substrs=["size=0", "{}"])
        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect(
            "frame variable ii",
            substrs=[
                "size=6",
                "[0] = 0",
                "[1] = 1",
                "[2] = 2",
                "[3] = 3",
                "[4] = 4",
                "[5] = 5",
            ],
        )
        lldbutil.continue_to_breakpoint(process, bkpt)

        self.check("ii", 7)

        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect("frame variable ii", substrs=["size=0", "{}"])
        self.check("ii", 0)
        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect("frame variable ii", substrs=["size=0", "{}"])
        ss_type = self.getVariableType("ss")
        self.assertTrue(
            ss_type.startswith(self.namespace + "::multiset"), "Type: " + ss_type
        )
        self.expect("frame variable ss", substrs=["size=0", "{}"])
        self.check("ss", 0)
        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect(
            "frame variable ss",
            substrs=["size=2", '[0] = "a"', '[1] = "a very long string is right here"'],
        )
        self.check("ss", 2)
        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect(
            "frame variable ss",
            substrs=[
                "size=4",
                '[0] = "a"',
                '[1] = "a very long string is right here"',
                '[2] = "b"',
                '[3] = "c"',
            ],
        )
        self.check("ss", 4)
        self.expect(
            "expression ss",
            substrs=[
                "size=4",
                '[0] = "a"',
                '[1] = "a very long string is right here"',
                '[2] = "b"',
                '[3] = "c"',
            ],
        )
        self.expect("frame variable ss[2]", substrs=[' = "b"'])
        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect(
            "frame variable ss",
            substrs=[
                "size=3",
                '[0] = "a"',
                '[1] = "a very long string is right here"',
                '[2] = "c"',
            ],
        )

    def do_test_ref_and_ptr(self):
        """Test that the data formatters work on ref and ptr."""
        (self.target, process, _, bkpt) = lldbutil.run_to_source_breakpoint(
            self,
            "Stop here to check by ref and ptr.",
            lldb.SBFileSpec("main.cpp", False),
        )
        # The reference should print just like the value:
        self.check("ref", 7)

        self.expect("frame variable ptr", substrs=["ptr =", "size=7"])
        self.expect("expr ptr", substrs=["size=7"])

    @add_test_categories(["libstdcxx"])
    def test_with_run_command_libstdcpp(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test_with_run_command()

    @add_test_categories(["libc++"])
    def test_with_run_command_libcpp(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test_with_run_command()

    @add_test_categories(["msvcstl"])
    def test_with_run_command_msvcstl(self):
        # No flags, because the "msvcstl" category checks that the MSVC STL is used by default.
        self.build()
        self.do_test_with_run_command()

    @add_test_categories(["libstdcxx"])
    def test_ref_and_ptr_libstdcpp(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test_ref_and_ptr()

    @add_test_categories(["libc++"])
    def test_ref_and_ptr_libcpp(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test_ref_and_ptr()

    @add_test_categories(["msvcstl"])
    def test_ref_and_ptr_msvcstl(self):
        self.build()
        self.do_test_ref_and_ptr()
