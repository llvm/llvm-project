# TestSwiftPrivateTypeAlias.py
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
Test that we correctly find private decls
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftPrivateTypeAlias(TestBase):
    @swiftTest
    def test_swift_private_typealias(self):
        """Test that we can correctly print variables whose types are private type aliases"""
        self.build()
        (target, process, thread, breakpoint1) = \
            lldbutil.run_to_source_breakpoint(
                self, 'breakpoint 1', lldb.SBFileSpec('main.swift'))
        breakpoint2 = target.BreakpointCreateBySourceRegex(
            'breakpoint 2', lldb.SBFileSpec('main.swift'))
        self.assertTrue(breakpoint1.GetNumLocations() > 0, VALID_BREAKPOINT)
        self.assertTrue(breakpoint2.GetNumLocations() > 0, VALID_BREAKPOINT)

        var = self.frame().FindVariable("i")
        lldbutil.check_variable(
            self,
            var,
            False,
            typename="a.MyStruct.IntegerType",
            value="123")

        process.Continue()
        threads = lldbutil.get_threads_stopped_at_breakpoint(process,
                                                             breakpoint2)
        self.assertTrue(len(threads) == 1)

        var = self.frame().FindVariable("a")
        dict_child_0 = var.GetChildAtIndex(0)
        child_0 = dict_child_0.GetChildAtIndex(0)
        child_1 = dict_child_0.GetChildAtIndex(1)
        lldbutil.check_variable(
            self, var, False, typename=
            "Swift.Dictionary<Swift.String, a.MyStruct.IntegerType>")
        lldbutil.check_variable(self, child_0, False, '"hello"')
        lldbutil.check_variable(self, child_1, False, value='234')

