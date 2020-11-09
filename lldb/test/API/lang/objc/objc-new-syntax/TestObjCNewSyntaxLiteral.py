"""Test that the Objective-C syntax for dictionary/array literals and indexing works"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from ObjCNewSyntaxTest import ObjCNewSyntaxTest


class ObjCNewSyntaxTestCaseLiteral(ObjCNewSyntaxTest):

    @skipIf(macos_version=["<", "10.12"])
    @expectedFailureAll(archs=["i[3-6]86"])
    def test_char_literal(self):
        self.runToBreakpoint()

        self.expect("expr --object-description -- @'a'",
                    VARIABLES_DISPLAYED_CORRECTLY, substrs=[str(ord('a'))])

    @skipIf(macos_version=["<", "10.12"])
    @expectedFailureAll(archs=["i[3-6]86"])
    def test_integer_literals(self):
        self.runToBreakpoint()

        self.expect(
            "expr --object-description -- @1",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["1"])

        self.expect(
            "expr --object-description -- @1l",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["1"])

        self.expect(
            "expr --object-description -- @1ul",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["1"])

        self.expect(
            "expr --object-description -- @1ll",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["1"])

        self.expect(
            "expr --object-description -- @1ull",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["1"])

    @skipIf(macos_version=["<", "10.12"])
    @expectedFailureAll(archs=["i[3-6]86"])
    def test_float_literal(self):
        self.runToBreakpoint()

        self.expect("expr -- @123.45", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["NSNumber", "123.45"])

    @skipIf(macos_version=["<", "10.12"])
    @expectedFailureAll(archs=["i[3-6]86"])
    def test_expressions_in_literals(self):
        self.runToBreakpoint()

        self.expect(
            "expr --object-description -- @( 1 + 3 )",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["4"])
        self.expect(
            "expr -- @((char*)\"Hello world\" + 6)",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "NSString",
                "world"])
