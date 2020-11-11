"""Test that the Objective-C syntax for dictionary/array literals and indexing works"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from ObjCNewSyntaxTest import ObjCNewSyntaxTest


class ObjCNewSyntaxTestCaseDictionary(ObjCNewSyntaxTest):

    @skipIf(macos_version=["<", "10.12"])
    @expectedFailureAll(archs=["i[3-6]86"])
    def test_read_dictionary(self):
        self.runToBreakpoint()

        self.expect(
            "expr --object-description -- immutable_dictionary[@\"key\"]",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["value"])

        self.expect(
            "expr --object-description -- mutable_dictionary[@\"key\"]",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["value"])

    @skipIf(macos_version=["<", "10.12"])
    @expectedFailureAll(archs=["i[3-6]86"])
    def test_update_dictionary(self):
        self.runToBreakpoint()

        self.expect(
            "expr --object-description -- mutable_dictionary[@\"key\"] = @\"object\"",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["object"])

        self.expect(
            "expr --object-description -- mutable_dictionary[@\"key\"]",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["object"])

    @skipIf(macos_version=["<", "10.12"])
    @expectedFailureAll(archs=["i[3-6]86"])
    def test_dictionary_literal(self):
        self.runToBreakpoint()

        self.expect(
            "expr --object-description -- @{ @\"key\" : @\"object\" }",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "key",
                "object"])
