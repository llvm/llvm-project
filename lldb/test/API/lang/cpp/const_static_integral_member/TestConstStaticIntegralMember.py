"""
Tests const static data members as specified by C++11 [class.static.data]p3.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here",
                                          lldb.SBFileSpec("main.cpp"))

        # Test using a simple const static integer member.
        self.expect_expr("A::int_val", result_value="1")

        # Try accessing the int member via some expressions that still produce
        # an lvalue.
        self.expect_expr("a.int_val", result_value="1")
        self.expect_expr("(A::int_val)", result_value="1")
        self.expect_expr("+A::int_val", result_value="1")
        self.expect_expr("1,A::int_val", result_value="1")
        self.expect_expr("true ? A::int_val : A::int_val", result_value="1")

        # Test a simple integer member that was also defined in a namespace
        # scope and has an address.
        self.expect_expr("A::int_val_with_address", result_value="2")

        # Test a bool member.
        self.expect_expr("A::bool_val", result_value="true")

        # Test that minimum and maximum values for each data type are right.
        self.expect_expr("A::char_max == char_max", result_value="true")
        self.expect_expr("A::uchar_max == uchar_max", result_value="true")
        self.expect_expr("A::int_max == int_max", result_value="true")
        self.expect_expr("A::uint_max == uint_max", result_value="true")
        self.expect_expr("A::long_max == long_max", result_value="true")
        self.expect_expr("A::ulong_max == ulong_max", result_value="true")
        self.expect_expr("A::longlong_max == longlong_max", result_value="true")
        self.expect_expr("A::ulonglong_max == ulonglong_max", result_value="true")

        self.expect_expr("A::char_min == char_min", result_value="true")
        self.expect_expr("A::uchar_min == uchar_min", result_value="true")
        self.expect_expr("A::int_min == int_min", result_value="true")
        self.expect_expr("A::uint_min == uint_min", result_value="true")
        self.expect_expr("A::long_min == long_min", result_value="true")
        self.expect_expr("A::ulong_min == ulong_min", result_value="true")
        self.expect_expr("A::longlong_min == longlong_min", result_value="true")
        self.expect_expr("A::ulonglong_min == ulonglong_min", result_value="true")

        # Test an unscoped enum.
        self.expect_expr("A::enum_val", result_value="enum_case2")
        # Test an unscoped enum with an invalid enum case.
        self.expect_expr("A::invalid_enum_val", result_value="enum_case1 | enum_case2 | 0x4")

        # Test a scoped enum.
        self.expect_expr("A::scoped_enum_val", result_value="scoped_enum_case2")
        # Test an scoped enum with an invalid enum case.
        self.expect_expr("A::invalid_scoped_enum_val", result_value="scoped_enum_case1 | 0x4")

        # Test an enum with fixed underlying type.
        self.expect_expr("A::scoped_char_enum_val", result_value="case2")
        self.expect_expr("A::scoped_ll_enum_val_neg", result_value="case0")
        self.expect_expr("A::scoped_ll_enum_val", result_value="case2")

        # Test taking address.
        if lldbplatformutil.getPlatform() == "windows":
            # On Windows data members without the out-of-class definitions still have
            # valid adresses and the following expression works fine.
            self.expect_expr("const int *i = &A::int_val; *i", result_value="1")
        else:
            # On other platforms (Linux, macos) data members without the out-of-class
            # definitions don't have valid addresses and the following code produces
            # a linker error.
            self.expect("expr const int *i = &A::int_val; *i", error=True,
                        substrs=["Couldn't lookup symbols:"])

        # This should work on all platforms.
        self.expect_expr("const int *i = &A::int_val_with_address; *i",
                         result_value="2")

    # dsymutil strips the debug info for classes that only have const static
    # data members without a definition namespace scope.
    @expectedFailureAll(debug_info=["dsym"])
    def test_class_with_only_const_static(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.cpp"))

        self.expect_expr("ClassWithOnlyConstStatic::member", result_value="3")

        # Test `constexpr static`.
        self.expect_expr("ClassWithConstexprs::member", result_value="2")
        self.expect_expr("ClassWithConstexprs::enum_val", result_value="enum_case2")
        self.expect_expr("ClassWithConstexprs::scoped_enum_val", result_value="scoped_enum_case2")
