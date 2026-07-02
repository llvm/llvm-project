"""
Test the SBTarget API's to retrieve persistent types
and values.
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestPersistentDecls(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_persistent_types(self):
        """Define some types in the expression evaluator and find them."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.types_test()

    def test_persistent_values(self):
        """Define some values in the expression evaluator and find them."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.values_test()

    def types_test(self):
        (target, _, _, _) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", self.main_source_file
        )

        # Make some types so we aren't succeeding in gettting the only one.

        typename = "$firstStruct"

        self.expect(f"expr struct $SomeType {{ int b; int a; int c;}}")
        self.expect(f"expr struct {typename} {{ int a; int b; char *c;}}")
        self.expect(f"expr struct $AnotherType {{ int a; char *b; int c;}}")
        self.expect(f"expr struct $YetAnotherType {{ char *a; int b; int c;}}")

        language = lldb.eLanguageTypeC_plus_plus
        error = lldb.SBError()
        type = target.FindExpressionTypeForLanguage(typename, language, error)
        self.assertSuccess(error, "Got the right error")
        self.assertTrue(type.IsValid(), "Got a valid type")

        # Make sure we got the type we expected:
        self.assertEqual(type.name, typename, "Got the right type.")
        # Now check that the fields are right:
        self.assertEqual(type.num_fields, 3, "Right number of fields")

        ivar_a = type.fields[0]
        self.assertEqual(ivar_a.name, "a", "a name is right")
        self.assertEqual(ivar_a.type.name, "int", "Right type")

        ivar_b = type.fields[1]
        self.assertEqual(ivar_b.name, "b", "b name is right")
        self.assertEqual(ivar_b.type.name, "int", "Right type")

        ivar_c = type.fields[2]
        self.assertEqual(ivar_c.name, "c", "c name is right")
        self.assertEqual(ivar_c.type.name, "char *", "Right type")

        # Also test the error returns:
        type = target.FindExpressionTypeForLanguage(
            typename, lldb.eLanguageTypeUnknown, error
        )
        self.assertFalse(error.Success(), "unknown doesn't support expression types")
        self.assertFalse(type.IsValid(), "Type is also invalid.")

        type = target.FindExpressionTypeForLanguage("", language, error)
        self.assertFalse(error.Success(), "empty type name for search.")
        self.assertFalse(type.IsValid(), "Type is also invalid.")
        type = target.FindExpressionTypeForLanguage(
            "doesnt_start_with_dollar", language, error
        )
        self.assertFalse(error.Success(), "Can't find names that don't exist.")
        self.assertFalse(type.IsValid(), "Type is also invalid.")

    def values_test(self):
        (target, _, _, _) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", self.main_source_file
        )

        type_name = "struct $MyStruct"
        value_name = "$my_struct"
        self.expect("expr int $my_int = 100")
        self.expect('expr char *$my_char_ptr = "Some char pointer."')
        self.expect(
            f"expr {type_name} {{int a; int b;}}; {type_name} {value_name} = {{10, 20}}"
        )

        # Now try to find one:
        language = lldb.eLanguageTypeC_plus_plus

        value = target.FindExpressionVariableForLanguage(value_name, language)

        self.assertTrue(value.IsValid(), "Got a valid SBValue")
        self.assertSuccess(value.GetError(), "Got no errors")

        value_checker = ValueCheck(
            name=value_name,
            type=type_name,
            children=[
                ValueCheck(name="a", type="int", value="10"),
                ValueCheck(name="b", type="int", value="20"),
            ],
        )
        value_checker.check_value(self, value, "Found the right value")

        # Also test that errors are set correctly:
        value = target.FindExpressionVariableForLanguage(
            value_name, lldb.eLanguageTypeUnknown
        )
        self.assertFalse(value.error.Success(), "unknown can't compile expressions")
        value = target.FindExpressionVariableForLanguage(None, language)
        self.assertFalse(value.error.Success(), "reject empty expression variable name")
        value = target.FindExpressionVariableForLanguage(
            "doesnt_start_with_dollar", language
        )
        self.assertFalse(value.error.Success(), "error on unknown variable name")
