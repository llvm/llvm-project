# TestSwiftGenericTupleLabels.py
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
Test that LLDB can reconstruct tuple labels from metadata
"""
import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftGenericTupleLabels(lldbtest.TestBase):

    def setUp(self):
        lldbtest.TestBase.setUp(self)

    @swiftTest
    def test_generic_tuple_labels(self):
        """Test that LLDB can reconstruct tuple labels from metadata"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        the_tuple = self.frame().FindVariable('x')
        the_tuple.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        the_tuple.SetPreferSyntheticValue(True)

        self.assertTrue(the_tuple.GetChildAtIndex(
            0).GetName() == 'x', '.0 == x')
        self.assertTrue(the_tuple.GetChildAtIndex(
            1).GetName() == '1', '.1 == 1')
        self.assertTrue(the_tuple.GetChildAtIndex(
            2).GetName() == 'z', '.2 == z')
        self.assertTrue(the_tuple.GetChildAtIndex(
            3).GetName() == '3', '.3 == 3')
        self.assertTrue(the_tuple.GetChildAtIndex(
            4).GetName() == 'q', '.4 == q')
        self.assertTrue(the_tuple.GetChildAtIndex(
            5).GetName() == 'w', '.5 == q')

        self.expect('frame variable -d run -- x.w', substrs=['72'])
        self.expect('expression -d run -- x.z', substrs=['36'])

