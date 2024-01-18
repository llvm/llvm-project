# TestSwiftAssociatedTypeResolution.py
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
Test that associated-typed objects get resolved to their proper location in memory
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftArchetypeResolution(TestBase):
    @swiftTest
    def test_swift_associated_type_resolution(self):
        """Test that archetype-typed objects get resolved to their proper location in memory"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set a breakpoint here', lldb.SBFileSpec('main.swift'))

        var = self.frame().FindVariable("things")
        var.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        var.SetPreferSyntheticValue(True)
        self.assertSuccess(var.GetError(), "Failed to get things")
        self.assertEqual(var.GetNumChildren(), 4,
                         "Got the right number of children")
        type_name = var.GetTypeName()
        self.assertEqual(type_name, "Swift.Array<Swift.Int>",
                         "Wrong typename: %s."%(type_name))
        for i in range(0,4):
            child = var.GetChildAtIndex(i)
            self.assertSuccess(child.GetError(), "Failed to get things[%d]" % i)
            value = child.GetValueAsUnsigned()
            self.assertEqual(value, i, "Wrong value: %d not %d."%(value, i))
