# TestSimpleExpressions.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Tests simple swift expressions
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import sys
import unittest2


class TestSimpleSwiftExpressions(TestBase):
    @swiftTest
    def test_simple_swift_expressions(self):
        """Tests that we can run simple Swift expressions correctly"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        # Test that parse errors give a correct result:
        value_obj = self.frame().EvaluateExpression(
            "iff is_five === 5 { return is_five")
        error = value_obj.GetError()

        # Test simple math with constants

        lldbutil.check_expression(self, self.frame(), "5 + 6", "11", use_summary=False)
        lldbutil.check_expression(self, self.frame(), "is_five + is_six", "11", use_summary=False)
        lldbutil.check_expression(self, self.frame(),
            "if (1 == 1) { return is_five + is_six }",
            "11",
            use_summary=False)

        # Test boolean operations with simple variables:
        # Bool's are currently enums, so their value is actually in the value.
        lldbutil.check_expression(self, self.frame(), "is_eleven == is_five + is_six", "true")

        # Try a slightly more complex container for our expression:
        lldbutil.check_expression(self, self.frame(),
            "if is_five == 5 { return is_five + is_six } else { return is_five }",
            "11",
            use_summary=False)

        # Make sure we get an error if we don't give homogenous return types:
        bool_or_int = self.frame().EvaluateExpression(
            "if is_five == 5 { return is_five + is_six } else { return false }")
        self.assertTrue(
            bool_or_int.IsValid(),
            "if is_five == 5 { return is_five + is_six } else { return false } is invalid")

        # Make sure we get the correct branch of a complex result expression:
        lldbutil.check_expression(self, self.frame(),
            "if is_five == 6 {return is_five} else if is_six == 5 {return is_six} ; is_eleven",
            "11",
            use_summary=False)

        # Make sure we can access globals:
        # Commented out till we resolve <rdar://problem/15695494> Accessing global variables causes LLVM ERROR and exit...
        # lldbutil.check_expression(self, self.frame(), "my_global", "30", use_summary=True)

        # Non-simple names:
        # Note: python 2 and python 3 have different default encodings.
        # This can be removed once python 2 is gone entirely.
        if sys.version_info.major == 2:
            lldbutil.check_expression(self, self.frame(),
                u"\u20ac_varname".encode("utf-8"),
                "5",
                use_summary=False)
        else:
            lldbutil.check_expression(self, self.frame(),
                u"\u20ac_varname",
                "5",
                use_summary=False)

        # See if we can do the same manipulations with tuples:
        # Commented out due to: <rdar://problem/15476525> Expressions with
        # tuple elements assert
        lldbutil.check_expression(self, self.frame(), "a_tuple.0 + a_tuple.1", "11", use_summary=False)

        # See if we can do some manipulations with dicts:
        lldbutil.check_expression(self, self.frame(),
            'str_int_dict["five"]! + str_int_dict["six"]!',
            "11",
            use_summary=False)
        lldbutil.check_expression(self, self.frame(),
            'int_str_dict[Int(is_five + is_six)]!',
            '"eleven"')

        # Commented out, touching the dict twice causes it to die, probably the same problem
        # as <rdar://problem/15306399>
        lldbutil.check_expression(self, self.frame(),
            'str_int_dict["five"] = 6; str_int_dict["five"]! + str_int_dict["six"]!',
            "12",
            use_summary=False)

        # See if we can use a switch statement in an expression:
        lldbutil.check_expression(self, self.frame(),
            "switch is_five { case 0..<6: return 1; case 7..<11: return 2; case _: return 4; }; 3;",
            "1",
            use_summary=False)

        # These ones are int-convertible and Equatable so we can do some things
        # with them anyway:
        lldbutil.check_expression(self, self.frame(), "enum_eleven", "Eleven", use_summary=False)
        lldbutil.check_expression(self, self.frame(), "enum_eleven == SomeValues.Eleven", "true", use_summary=True)
        lldbutil.check_expression(self, self.frame(),
            "SomeValues.Five.toInt() + SomeValues.Six.toInt()",
            "11",
            use_summary=False)
        lldbutil.check_expression(self, self.frame(),
                                  "enum_eleven = .Five; return enum_eleven == .Five", "true", use_summary=True)

        # Test expressions with a simple object:
        lldbutil.check_expression(self, self.frame(), "a_obj.x", "6", use_summary=False)

        # Should not have to make a second object here. This is another
        # side-effect of <rdar://problem/15306399>
        lldbutil.check_expression(self, self.frame(), "a_nother_obj.y", "6.5", use_summary=False)

        # Test expressions with a struct:
        lldbutil.check_expression(self, self.frame(), "b_struct.b_int", "5", use_summary=False)

        # Test expression with Chars and strings:

        # Commented out due to <rdar://problem/16856379>
        #self.check_expression ("a_char", "U+0061 U+0000 u'a'")

        # Interpolated strings and string addition:
        lldbutil.check_expression(self, self.frame(),
            '"Five: \(is_five) " + "Six: \(is_six)"',
                                  '"Five: 5 Six: 6"', use_summary=True)

        # Next let's try some simple array accesses:
        lldbutil.check_expression(self, self.frame(), "an_int_array[0]", "5", use_summary=False)

        # Test expression with read-only variables:
        lldbutil.check_expression(self, self.frame(), "b_struct.b_read_only == 5", "true", use_summary=True)
        failed_value = self.frame().EvaluateExpression(
            "b_struct.b_read_only = 34")
        self.assertTrue(failed_value.IsValid(),
                        "Get something back from the evaluation.")
        self.assertTrue(failed_value.GetError().Success()
                        == False, "But it is an error.")

        # Check a simple value in a struct:
        lldbutil.check_expression(self, self.frame(), "b_struct_2.b_int", "20", use_summary=False)

        # Make sure this works for properties in extensions as well:
        lldbutil.check_expression(self, self.frame(), "b_struct_2.b_float", "20.5", use_summary=False)

        # Here are a few tests of making variables in expressions:
        lldbutil.check_expression(self, self.frame(),
            "var enum_six : SomeValues = SomeValues.Six; return enum_six == .Six", "true")
