"""Test that the Objective-C syntax for dictionary/array literals and indexing works"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from ObjCNewSyntaxTest import ObjCNewSyntaxTest


class ObjCNewSyntaxTestCaseArray(ObjCNewSyntaxTest):

    @skipIf(macos_version=["<", "10.12"])
    @expectedFailureAll(archs=["i[3-6]86"])
    def test_read_array(self):
        self.runToBreakpoint()

        self.expect(
            "expr --object-description -- immutable_array[0]",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["foo"])

        self.expect(
            "expr --object-description -- mutable_array[0]",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["foo"])

    @skipIf(macos_version=["<", "10.12"])
    @expectedFailureAll(archs=["i[3-6]86"])
    def test_update_array(self):
        self.runToBreakpoint()

        self.expect(
            "expr --object-description -- mutable_array[0] = @\"bar\"",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["bar"])

        self.expect(
            "expr --object-description -- mutable_array[0]",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["bar"])

    @skipIf(macos_version=["<", "10.12"])
    @expectedFailureAll(archs=["i[3-6]86"])
    def test_array_literal(self):
        self.runToBreakpoint()

        self.expect(
            "expr --object-description -- @[ @\"foo\", @\"bar\" ]",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "NSArray",
                "foo",
                "bar"])
