"""Look up enum type information and check for correct display."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class CPP11EnumTypesTestCase(TestBase):

    def check_enum(self, suffix):
        """
        :param suffix The suffix of the enum type name (enum_<suffix>) that
                      should be checked.
        :param test_values A list of integet values that shouldn't be converted
                           to any valid enum case.
        """
        enum_name = "enum_" + suffix
        unsigned = suffix.startswith("u")

        self.expect("image lookup -t " + enum_name,
                    patterns=["enum( struct| class)? " + enum_name + " {"],
                    substrs=["Case1",
                             "Case2",
                             "Case3"])
        # Test each case in the enum.
        self.expect_expr("var1_" + suffix, result_type=enum_name, result_value="Case1")
        self.expect_expr("var2_" + suffix, result_type=enum_name, result_value="Case2")
        self.expect_expr("var3_" + suffix, result_type=enum_name, result_value="Case3")

        if unsigned:
            self.expect_expr("var_below_" + suffix, result_type=enum_name, result_value="199")
            self.expect_expr("var_above_" + suffix, result_type=enum_name, result_value="203")
        else:
            self.expect_expr("var_below_" + suffix, result_type=enum_name, result_value="-3")
            self.expect_expr("var_above_" + suffix, result_type=enum_name, result_value="1")

    @skipIf(dwarf_version=['<', '4'])
    def test(self):
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.check_enum("uc")
        self.check_enum("c")
        self.check_enum("us")
        self.check_enum("s")
        self.check_enum("ui")
        self.check_enum("i")
        self.check_enum("ul")
        self.check_enum("l")
        self.check_enum("ull")
        self.check_enum("ll")
