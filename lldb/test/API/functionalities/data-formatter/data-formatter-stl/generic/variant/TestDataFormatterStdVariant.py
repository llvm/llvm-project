"""
Test lldb data formatter subsystem.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdVariantDataFormatterTestCase(TestBase):
    def do_test(self):
        """Test that that file and class static variables display correctly."""

        def cleanup():
            self.runCmd("type format clear", check=False)
            self.runCmd("type summary clear", check=False)
            self.runCmd("type filter clear", check=False)
            self.runCmd("type synth clear", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        (self.target, self.process, _, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        for name in ["v1", "v1_typedef"]:
            self.expect(
                "frame variable " + name,
                substrs=[name + " =  Active Type = int  {", "Value = 12", "}"],
            )

        for name in ["v1_ref", "v1_typedef_ref"]:
            self.expect(
                "frame variable " + name,
                patterns=[name + " = 0x.*  Active Type = int : {", "Value = 12", "}"],
            )

        self.expect(
            "frame variable v_v1",
            substrs=[
                "v_v1 =  Active Type = std::variant<int, double, char>  {",
                "Value =  Active Type = int  {",
                "Value = 12",
                "}",
                "}",
            ],
        )

        lldbutil.continue_to_breakpoint(self.process, bkpt)

        self.expect(
            "frame variable v1",
            substrs=["v1 =  Active Type = double  {", "Value = 2", "}"],
        )

        lldbutil.continue_to_breakpoint(self.process, bkpt)

        self.expect(
            "frame variable v2",
            substrs=["v2 =  Active Type = double  {", "Value = 2", "}"],
        )

        self.expect(
            "frame variable v3",
            substrs=["v3 =  Active Type = char  {", "Value = 'A'", "}"],
        )

        self.expect("frame variable v_valueless", substrs=["v_valueless =  No Value"])

        self.expect(
            "frame variable v_300_types_valueless",
            substrs=["v_300_types_valueless =  No Value"],
        )

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
