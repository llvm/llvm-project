"""
Test types declared in various kinds of scopes (global, classes, etc.).
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


def names(name, display=None, canonical=None):
    """The expected (GetName, GetDisplayTypeName, GetCanonicalType().GetName())
    of a type. The display and canonical names default to the name."""
    return (name, display or name, canonical or name)


EXPECTED = {
    # Global scope.
    "g_global_struct": names("GlobalStruct"),
    "g_global_enum": names("GlobalEnum"),
    "g_global_typedef": names("GlobalTypedef", canonical="GlobalStruct"),
    "g_global_unnamed_struct": names("(unnamed struct)"),
    # extern "C" is transparent.
    "g_extern_c_struct": names("ExternCStruct"),
    # Namespaces.
    "g_in_ns": names("ns::InNs"),
    "g_ns_scoped_enum": names("ns::ScopedEnum"),
    "g_ns_typedef": names("ns::NsTypedef", canonical="int"),
    "g_ns_alias": names("ns::NsAlias", canonical="ns::InNs"),
    "g_unnamed_in_ns": names("ns::(unnamed struct)"),
    "g_in_nested_ns": names("ns::inner::InNested"),
    "g_in_inline_ns": names("ns::inl::InInline", display="ns::InInline"),
    "g_in_anon_in_ns": names(
        "ns::(anonymous namespace)::InAnonInNs", display="ns::InAnonInNs"
    ),
    "g_in_anon": names("(anonymous namespace)::InAnon", display="InAnon"),
    # Records.
    "g_outer_inner": names("Outer::Inner"),
    "g_outer_inner_enum": names("Outer::InnerEnum"),
    "g_outer_member_typedef": names("Outer::MemberTypedef", canonical="int"),
    "g_outer_unnamed_member": names("Outer::(unnamed struct)"),
    "g_outer_innermost": names("Outer::Inner2::Innermost"),
    "g_ns_outer_inner": names("ns::NsOuter::Inner"),
    "g_tmpl": names("Tmpl<int>"),
    "g_tmpl_nested": names("Tmpl<int>::Nested"),
    "g_tmpl_nested_tmpl": names("Tmpl<int>::NestedTmpl<char>"),
    "g_ns_tmpl_nested": names("ns::NsTmpl<Outer::Inner>::Nested"),
    # Types whose direct context is a function: no scope at all.
    "g_func_local": names("Local"),
    "g_func_local_enum": names("LocalEnum"),
    "g_func_local_typedef": names("LocalTypedef", canonical="int"),
    "g_func_unnamed_local": names("(unnamed struct)"),
    "g_block_local": names("InBlock"),
    "g_ns_func_local": names("NsLocal"),
    "g_static_method_local": names("InStatic"),
    "g_tmpl_func_local": names("TLocal"),
    "g_in_lambda": names("InLambda"),
    # Types nested in a local class: the function is part of the scope.
    "g_func_nested_local": names("global_func()::A_func::B"),
    "g_func_nested_unnamed_local": names("global_func()::A_func::(unnamed struct)"),
    # A lexical block is not a scope of its own.
    "g_block_nested_local": names("global_func()::A_block::B"),
    "g_nested_local_int_char": names("ns::nested_local_func(int, char)::A_int_char::B"),
    "g_nested_local_double": names("ns::nested_local_func(double)::A_double::B"),
    "g_nested_local_variadic": names(
        "ns::nested_local_variadic(int, ...)::A_variadic::B"
    ),
    "g_nested_local_anon_ns": names(
        "ns::(anonymous namespace)::nested_local_anon_ns()::A_anon_ns::B",
        display="ns::nested_local_anon_ns()::A_anon_ns::B",
    ),
    "g_nested_local_inline_ns": names(
        "ns::inl::nested_local_inline_ns()::A_inline_ns::B",
        display="ns::nested_local_inline_ns()::A_inline_ns::B",
    ),
    "g_nested_local_extern_c": names("nested_local_extern_c()::A_extern_c::B"),
    # The cv-qualifier of a method is not part of the scope.
    "g_method_nested_local": names("WithMethods::method(int)::A_method::B"),
    "g_ctor_nested_local": names("WithMethods::WithMethods()::A_ctor::B"),
    "g_conversion_nested_local": names("WithMethods::operator int()::A_conversion::B"),
    "g_tmpl_class_method_nested_local": names(
        "ns::TmplWithMethod<Outer::Inner>::method(Outer::Inner)"
        "::A_tmpl_class_method::B"
    ),
    # Lambdas. The closure type is unnamed in DWARF.
    "g_global_lambda": names("(unnamed class)"),
    "g_ns_lambda": names("ns::(unnamed class)"),
    "g_func_lambda": names("(unnamed class)"),
    "g_nested_in_lambda": names("(unnamed class)::operator()()::A_in_lambda::B"),
    "g_nested_in_ns_lambda": names(
        "ns::(unnamed class)::operator()(int)::A_in_ns_lambda::B"
    ),
}

# The template arguments of a function template specialization are not
# part of the scope, only its parameter types are.
EXPECTED_TMPL_FUNC = {
    "g_tmpl_func_nested_local": names("tmpl_func(int)::A_tmpl_func::B"),
}

# Same-named local classes in different functions are different scopes.
EXPECTED_COLLIDING = {
    "g_collide1": names("collide1()::Same::B"),
    "g_collide2": names("collide2()::Same::B"),
}


class TestCppTypeScopeNames(TestBase):
    def collect_static_locals(self, target, block, result):
        """
        Collect the g_* variables of `block` and all its nested blocks.
        This is needed as we can't find the static locals via globals or an
        equivalent SB API.
        """
        while block.IsValid():
            for var in block.GetVariables(target, True, True, True):
                if var.GetName() and var.GetName().startswith("g_"):
                    result[var.GetName()] = var.GetType()
            self.collect_static_locals(target, block.GetFirstChild(), result)
            block = block.GetSibling()

    def find_types(self, target, expected_types):
        """Map the name of every g_* variable to its SBType."""
        exe_module = target.FindModule(target.GetExecutable())
        types = {}
        # Function-local statics are not in the global variable index, so walk
        # the blocks of all functions in the executable.
        for sc in target.FindGlobalFunctions(".", 0, lldb.eMatchTypeRegex):
            if sc.GetModule() != exe_module:
                continue
            self.collect_static_locals(target, sc.GetFunction().GetBlock(), types)
        for name in expected_types:
            if name not in types:
                var = target.FindFirstGlobalVariable(name)
                if var.IsValid():
                    types[name] = var.GetType()
        return types

    def check_types(self, expected_types):
        self.build()
        target = self.createTestTarget()
        types = self.find_types(target, expected_types)

        # Collect all mismatches so a failure shows the complete picture.
        mismatches = []
        for var_name, expected in expected_types.items():
            sbtype = types.get(var_name)
            if sbtype is None or not sbtype.IsValid():
                mismatches.append("%s: variable/type not found" % var_name)
                continue
            actual = (
                sbtype.GetName(),
                sbtype.GetDisplayTypeName(),
                sbtype.GetCanonicalType().GetName(),
            )
            for kind, exp, act in zip(
                ("name", "display name", "canonical name"), expected, actual
            ):
                if exp != act:
                    mismatches.append(
                        "%s: %s is '%s', expected '%s'" % (var_name, kind, act, exp)
                    )
        self.assertEqual(mismatches, [], "\n" + "\n".join(mismatches))

    @no_debug_info_test
    def test(self):
        self.check_types(EXPECTED)

    # FIXME: On Windows the template arguments are included in the scope:
    # 'tmpl_func<int>(int)::A_tmpl_func::B'.
    @expectedFailureAll(oslist=["windows"])
    @no_debug_info_test
    def test_tmpl_func_scope(self):
        self.check_types(EXPECTED_TMPL_FUNC)

    # FIXME: LLDB merges same-named local classes from different functions, so
    # g_collide2 is reported as 'collide1()::Same::B'.
    @expectedFailureAll
    @no_debug_info_test
    def test_colliding_local_classes(self):
        self.check_types(EXPECTED_COLLIDING)
