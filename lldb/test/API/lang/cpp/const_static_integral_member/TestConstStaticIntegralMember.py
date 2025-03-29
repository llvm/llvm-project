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
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

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
        self.expect_expr("A::schar_max == schar_max", result_value="true")
        self.expect_expr("A::uchar_max == uchar_max", result_value="true")
        self.expect_expr("A::int_max == int_max", result_value="true")
        self.expect_expr("A::uint_max == uint_max", result_value="true")
        self.expect_expr("A::long_max == long_max", result_value="true")
        self.expect_expr("A::ulong_max == ulong_max", result_value="true")
        self.expect_expr("A::longlong_max == longlong_max", result_value="true")
        self.expect_expr("A::ulonglong_max == ulonglong_max", result_value="true")
        self.expect_expr("A::wchar_max == wchar_max", result_value="true")

        self.expect_expr("A::char_min == char_min", result_value="true")
        self.expect_expr("A::schar_min == schar_min", result_value="true")
        self.expect_expr("A::uchar_min == uchar_min", result_value="true")
        self.expect_expr("A::int_min == int_min", result_value="true")
        self.expect_expr("A::uint_min == uint_min", result_value="true")
        self.expect_expr("A::long_min == long_min", result_value="true")
        self.expect_expr("A::ulong_min == ulong_min", result_value="true")
        self.expect_expr("A::longlong_min == longlong_min", result_value="true")
        self.expect_expr("A::ulonglong_min == ulonglong_min", result_value="true")
        self.expect_expr("A::wchar_min == wchar_min", result_value="true")

        # Test an unscoped enum.
        self.expect_expr("A::enum_val", result_value="enum_case2")
        # Test an unscoped enum with bool as the underlying type.
        self.expect_expr("A::enum_bool_val", result_value="enum_bool_case1")

        # Test a scoped enum.
        self.expect_expr("A::scoped_enum_val", result_value="scoped_enum_case2")
        # Test an scoped enum with a value that isn't an enumerator.
        self.expect_expr(
            "A::not_enumerator_scoped_enum_val", result_value="scoped_enum_case1 | 0x4"
        )
        # This time with more than one enum value plus the extra.
        self.expect_expr(
            "A::not_enumerator_scoped_enum_val_2",
            result_value="scoped_enum_case1 | scoped_enum_case2 | 0x4",
        )

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
            self.expect(
                "expr const int *i = &A::int_val; *i",
                error=True,
                substrs=["Couldn't look up symbols:"],
            )

        # This should work on all platforms.
        self.expect_expr(
            "const int *i = &A::int_val_with_address; *i", result_value="2"
        )

        # Printing the whole type takes a slightly different code path. Check that
        # it does not crash.
        self.expect("image lookup -t A")

    # dsymutil strips the debug info for classes that only have const static
    # data members without locations.
    @expectedFailureAll(debug_info=["dsym"], dwarf_version=["<", "5"])
    def test_class_with_only_const_static(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect_expr("ClassWithOnlyConstStatic::member", result_value="3")

    def check_global_var(self, name: str, expect_type, expect_val):
        var_list = self.target().FindGlobalVariables(name, lldb.UINT32_MAX)
        self.assertGreaterEqual(len(var_list), 1)
        varobj = var_list[0]
        self.assertEqual(varobj.type.name, expect_type)
        self.assertEqual(varobj.value, expect_val)

    def check_inline_static_members(self, flags):
        self.build(dictionary={"CXXFLAGS_EXTRAS": flags})
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        self.check_global_var("A::int_val", "const int", "1")
        self.check_global_var("A::int_val_with_address", "const int", "2")
        self.check_global_var("A::inline_int_val", "const int", "3")
        self.check_global_var("A::bool_val", "const bool", "true")
        self.check_global_var("A::enum_val", "Enum", "enum_case2")
        self.check_global_var("A::enum_bool_val", "EnumBool", "enum_bool_case1")
        self.check_global_var("A::scoped_enum_val", "ScopedEnum", "scoped_enum_case2")

        self.check_global_var("ClassWithOnlyConstStatic::member", "const int", "3")

        self.check_global_var("ClassWithConstexprs::member", "const int", "2")
        self.check_global_var("ClassWithConstexprs::enum_val", "Enum", "enum_case2")
        self.check_global_var(
            "ClassWithConstexprs::scoped_enum_val", "ScopedEnum", "scoped_enum_case2"
        )

    # Fails on Windows for unknown reasons.
    @skipIfWindows
    # On linux this passes due to the manual index
    @expectedFailureDarwin(debug_info=no_match(["dsym"]))
    @skipIf(debug_info=["dsym"], compiler=["clang"], compiler_version=["<", "19.0"])
    def test_inline_static_members_dwarf5(self):
        self.check_inline_static_members("-gdwarf-5")

    # On linux this passes due to the manual index
    @expectedFailureDarwin
    def test_inline_static_members_dwarf4(self):
        self.check_inline_static_members("-gdwarf-4")

    # With older versions of Clang, LLDB fails to evaluate classes with only
    # constexpr members when dsymutil is enabled
    @expectedFailureAll(
        debug_info=["dsym"], compiler=["clang"], compiler_version=["<", "14.0"]
    )
    def test_class_with_only_constexpr_static(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        # Test `constexpr static`.
        self.expect_expr("ClassWithConstexprs::member", result_value="2")
        self.expect_expr("ClassWithConstexprs::enum_val", result_value="enum_case2")
        self.expect_expr(
            "ClassWithConstexprs::scoped_enum_val", result_value="scoped_enum_case2"
        )

        # Test an aliased enum with fixed underlying type.
        self.expect_expr(
            "ClassWithEnumAlias::enum_alias", result_value="scoped_enum_case2"
        )
        self.expect_expr(
            "ClassWithEnumAlias::enum_alias_alias", result_value="scoped_enum_case1"
        )

    def check_shadowed_static_inline_members(self, flags):
        """Tests that the expression evaluator and SBAPI can both
        correctly determine the requested inline static variable
        in the presence of multiple variables of the same name."""

        self.build(dictionary={"CXXFLAGS_EXTRAS": flags})
        lldbutil.run_to_name_breakpoint(self, "bar")

        self.check_global_var("ns::Foo::mem", "const int", "10")

        self.expect_expr("mem", result_value="10")
        self.expect_expr("Foo::mem", result_value="10")
        self.expect_expr("ns::Foo::mem", result_value="10")
        self.expect_expr("::Foo::mem", result_value="-29")

    # Fails on Windows for unknown reasons.
    @skipIfWindows
    # On linux this passes due to the manual index
    @expectedFailureDarwin(debug_info=no_match(["dsym"]))
    @skipIf(debug_info=["dsym"], compiler=["clang"], compiler_version=["<", "19.0"])
    def test_shadowed_static_inline_members_dwarf5(self):
        self.check_shadowed_static_inline_members("-gdwarf-5")

    # On linux this passes due to the manual index
    @expectedFailureDarwin
    def test_shadowed_static_inline_members_dwarf4(self):
        self.check_shadowed_static_inline_members("-gdwarf-4")

    @expectedFailureAll(bugnumber="target var doesn't honour global namespace")
    def test_shadowed_static_inline_members_xfail(self):
        self.build()
        lldbutil.run_to_name_breakpoint(self, "bar")
        self.check_global_var("::Foo::mem", "const int", "-29")
