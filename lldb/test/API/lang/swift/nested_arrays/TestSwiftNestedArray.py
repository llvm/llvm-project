# TestSwiftNestedArray.py
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
Test Arrays of Arrays in Swift
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


def check_for_idx(child, idx):
    return child.GetValue() == str(idx + 1)


def check_for_C(child, idx):
    if child.GetTypeName() == "a.C":
        if child.GetNumChildren() == 1:
            if child.GetChildAtIndex(0).GetName() == "m_counter":
                return True
    return False


class TestSwiftNestedArray(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    @swiftTest
    def test_swift_nested_array(self):
        """Test Arrays of Arrays in Swift"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        var_aInt = self.frame().FindVariable("aInt")
        var_aC = self.frame().FindVariable("aC")
        lldbutil.check_variable(self, var_aInt, False, num_children=6)
        lldbutil.check_variable(self, var_aC, False, num_children=5)

        for i in range(0, 6):
            var_aIntChild = var_aInt.GetChildAtIndex(i)
            lldbutil.check_children(self, var_aIntChild, check_for_idx)

        for i in range(0, 5):
            var_aCChild = var_aC.GetChildAtIndex(i)
            lldbutil.check_children(self, var_aCChild, check_for_C)

