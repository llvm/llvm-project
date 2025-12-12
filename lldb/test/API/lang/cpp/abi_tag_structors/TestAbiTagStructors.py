"""
Test that we can call structors/destructors
annotated (and thus mangled) with ABI tags.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AbiTagStructorsTestCase(TestBase):
    @skipIf(
        compiler="clang",
        compiler_version=["<", "22"],
        bugnumber="Required Clang flag not supported",
    )
    @expectedFailureAll(oslist=["windows"])
    def test_with_structor_linkage_names(self):
        self.build(dictionary={"CXXFLAGS_EXTRAS": "-gstructor-decl-linkage-names"})

        lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec("main.cpp", False)
        )

        self.expect_expr(
            "Tagged()",
            result_type="Tagged",
            result_children=[ValueCheck(name="x", value="15")],
        )
        self.expect_expr(
            "Tagged(-17)",
            result_type="Tagged",
            result_children=[ValueCheck(name="x", value="-17")],
        )
        self.expect_expr("t1 = t2", result_type="Tagged")

        self.expect("expr Tagged t3(t1)", error=False)
        self.expect("expr t1.~Tagged()", error=False)

        self.expect("expr t1.~Tagged()", error=False)

        self.expect(
            "expression -- struct $Derived : virtual public Tagged { int y; $Derived(int val) : Tagged(val) { y = x; } };",
            error=False,
        )
        self.expect(
            "expression -- struct $Derived2 : virtual public $Derived { int z; $Derived2() : $Derived(10) { z = y; } };",
            error=False,
        )
        self.expect_expr(
            "$Derived2 d; d",
            result_type="$Derived2",
            result_children=[
                ValueCheck(
                    name="$Derived",
                    children=[
                        ValueCheck(
                            name="Tagged", children=[ValueCheck(name="x", value="15")]
                        ),
                        ValueCheck(name="y", value="15"),
                    ],
                ),
                ValueCheck(name="z", value="15"),
            ],
        )

        # Calls to deleting and base object destructor variants (D0 and D2 in Itanium ABI)
        self.expect_expr(
            "struct D : public HasVirtualDtor {}; D d; d.func()",
            result_type="int",
            result_value="10",
        )

    @expectedFailureAll(oslist=["windows"])
    def test_no_structor_linkage_names(self):
        """
        Test that without linkage names on structor declarations we can't call
        ABI-tagged structors.
        """
        # In older versions of Clang the -gno-structor-decl-linkage-names
        # behaviour was the default.
        if self.expectedCompiler(["clang"]) and self.expectedCompilerVersion(
            [">=", "22.0"]
        ):
            self.build(
                dictionary={"CXXFLAGS_EXTRAS": "-gno-structor-decl-linkage-names"}
            )
        else:
            self.build()

        lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec("main.cpp", False)
        )

        self.expect("expression Tagged(17)", error=True)
        self.expect("expr Tagged t3(t1)", error=True)
        self.expect("expr t1.~Tagged()", error=True)

        ## Calls to deleting and base object destructor variants (D0 and D2 in Itanium ABI)
        self.expect(
            "expression -- struct D : public HasVirtualDtor {}; D d; d.func()",
            error=True,
        )

        self.expect("expression -- Derived d(16); d", error=True)

    def do_nested_structor_test(self):
        """
        Test that calling ABI-tagged ctors of function local classes is not supported,
        but calling un-tagged functions is.
        """
        lldbutil.run_to_source_breakpoint(
            self, "Break nested", lldb.SBFileSpec("main.cpp", False)
        )

        self.expect("expression Local()", error=False)
        self.expect(
            "expression TaggedLocal()", error=True, substrs=["Couldn't look up symbols"]
        )

    @skipIf(compiler="clang", compiler_version=["<", "22"])
    @expectedFailureAll(oslist=["windows"])
    def test_nested_with_structor_linkage_names(self):
        self.build(dictionary={"CXXFLAGS_EXTRAS": "-gstructor-decl-linkage-names"})
        self.do_nested_structor_test()

    @expectedFailureAll(oslist=["windows"])
    def test_nested_no_structor_linkage_names(self):
        # In older versions of Clang the -gno-structor-decl-linkage-names
        # behaviour was the default.
        if self.expectedCompiler(["clang"]) and self.expectedCompilerVersion(
            [">=", "22.0"]
        ):
            self.build(
                dictionary={"CXXFLAGS_EXTRAS": "-gno-structor-decl-linkage-names"}
            )
        else:
            self.build()

        self.do_nested_structor_test()
