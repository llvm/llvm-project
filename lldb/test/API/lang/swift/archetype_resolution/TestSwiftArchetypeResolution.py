# TestSwiftArchetypeResolution.py
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
Test that archetype-typed objects get resolved to their proper location in memory
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftArchetypeResolution(TestBase):
    @swiftTest
    def test_swift_archetype_resolution(self):
        """Test that archetype-typed objects get resolved to their proper location in memory"""
        self.build()
        (target, process, thread, bkpt) = \
            lldbutil.run_to_source_breakpoint(
                self, 'break here', lldb.SBFileSpec('main.swift'))

        var_x = self.frame().FindVariable("x")
        var_x.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        var_x.SetPreferSyntheticValue(True)
        self.assertTrue(var_x.GetSummary() == '"hello"', "String case fails")
        if self.TraceOn():
            self.runCmd("frame variable -d run")

        process.Continue()
        var_x = self.frame().FindVariable("x")
        var_x.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        var_x.SetPreferSyntheticValue(True)
        self.assertTrue(var_x.GetValue() == '1', "Int case fails")
        if self.TraceOn():
            self.runCmd("frame variable -d run")

        process.Continue()
        var_x = self.frame().FindVariable("x")
        var_x.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        var_x.SetPreferSyntheticValue(True)
        self.assertTrue(var_x.GetChildMemberWithName(
            "y").GetSummary() == '"hello"', "S.String case fails")
        self.assertTrue(var_x.GetChildMemberWithName(
            "x").GetValue() == '1', "S.Int case fails")
        if self.TraceOn():
            self.runCmd("frame variable -d run")

        process.Continue()
        var_x = self.frame().FindVariable("x")
        var_x.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        var_x.SetPreferSyntheticValue(True)
        self.assertTrue(var_x.GetChildMemberWithName(
            "y").GetSummary() == '"hello"', "C.String case fails")
        self.assertTrue(var_x.GetChildMemberWithName(
            "x").GetValue() == '1', "C.Int case fails")
        if self.TraceOn():
            self.runCmd("frame variable -d run")

        process.Continue()
        var_x = self.frame().FindVariable("x")
        var_x.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        var_x.SetPreferSyntheticValue(True)
        self.assertTrue(var_x.GetChildMemberWithName(
            "0").GetValue() == '1', "Tuple.0 case fails")
        self.assertTrue(var_x.GetChildMemberWithName(
            "1").GetValue() == '2', "Tuple.1 case fails")
        if self.TraceOn():
            self.runCmd("frame variable -d run")

        process.Continue()
        var_x = self.frame().FindVariable("x")
        var_x.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        var_x.SetPreferSyntheticValue(True)
        self.assertTrue(var_x.GetValue() == 'A', "E case fails")
        if self.TraceOn():
            self.runCmd("frame variable -d run")

        process.Continue()
        var_x = self.frame().FindVariable("x")
        var_x.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        var_x.SetPreferSyntheticValue(True)
        self.assertTrue(var_x.GetChildMemberWithName(
            "y").GetSummary() == '"hello"', "GS.String case fails")
        self.assertTrue(var_x.GetChildMemberWithName(
            "x").GetValue() == '1', "GS.Int case fails")
        if self.TraceOn():
            self.runCmd("frame variable -d run")

        process.Continue()
        var_x = self.frame().FindVariable("x")
        var_x.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        var_x.SetPreferSyntheticValue(True)
        self.assertTrue(var_x.GetChildMemberWithName(
            "y").GetSummary() == '"hello"', "GC.String case fails")
        self.assertTrue(var_x.GetChildMemberWithName(
            "x").GetValue() == '1', "GC.Int case fails")
        if self.TraceOn():
            self.runCmd("frame variable -d run")

        process.Continue()
        var_x = self.frame().FindVariable("x")
        var_x.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        var_x.SetPreferSyntheticValue(True)
        self.assertTrue(var_x.GetValue() == 'A', "GE case fails")
        if self.TraceOn():
            self.runCmd("frame variable -d run")
