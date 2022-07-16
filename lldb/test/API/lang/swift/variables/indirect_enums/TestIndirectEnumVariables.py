# TestIndirectEnumVariables.py
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
Tests that indirect Enum variables display correctly
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestIndirectEnumVariables(TestBase):

    @swiftTest
    def test_indirect_cases_variables(self):
        """Tests that indirect Enum variables display correctly when cases are indirect"""
        self.build()
        self.do_test("indirect case break here")

    @swiftTest
    def test_indirect_enum_variables(self):
        """Tests that indirect Enum variables display correctly when enum is indirect"""
        self.build()
        self.do_test("indirect enum break here")

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def get_variable(self, name):
        x = self.frame().FindVariable(name)
        x.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        x.SetPreferSyntheticValue(True)
        return x

    def check_enum(
            self,
            enum,
            value=None,
            summary=None,
            child_path=None,
            child_value=None,
            child_summary=None):
        if value:
            self.assertTrue(
                enum.GetValue() == value, "%s.GetValue() == %s" %
                (enum.GetName(), value))
        if summary:
            self.assertTrue(
                enum.GetSummary() == summary, "%s.GetSummary() == %s" %
                (enum.GetName(), summary))

        if child_path:
            child = enum
            for child_index in child_path:
                child = child.GetChildAtIndex(child_index)
                child.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
                child.SetPreferSyntheticValue(True)
            self.assertTrue(
                child.IsValid(),
                "child at path %s valid" %
                (child_path))
            if child_value:
                self.assertTrue(
                    child.GetValue() == child_value, "%s.GetValue() == %s" %
                    (child.GetName(), child_value))
            if child_summary:
                self.assertTrue(
                    child.GetSummary() == child_summary, "%s.GetSummary() == %s" %
                    (child.GetName(), child_summary))

    def do_test(self, break_pattern):
        """Tests that indirect Enum variables display correctly"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            break_pattern, self.main_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)

        GP_StructType = self.get_variable("GP_StructType")
        GP_TupleType = self.get_variable("GP_TupleType")
        GP_ClassType = self.get_variable("GP_ClassType")
        GP_ProtocolType_Struct = self.get_variable("GP_ProtocolType_Struct")
        GP_ProtocolType_Class = self.get_variable("GP_ProtocolType_Class")
        GP_CEnumType = self.get_variable("GP_CEnumType")
        GP_ADTEnumType = self.get_variable("GP_ADTEnumType")
        GP_Recursive = self.get_variable("GP_Recursive")

        self.check_enum(
            GP_StructType,
            value='StructType',
            child_path=[0],
            child_value='12')

        self.check_enum(
            GP_TupleType,
            value='TupleType',
            child_path=[
                0,
                0],
            child_value='12')
        self.check_enum(
            GP_TupleType, value='TupleType', child_path=[
                0, 1], child_summary='"Hello World"')

        self.check_enum(
            GP_ClassType, value='ClassType', child_path=[
                0, 0, 0], child_summary='"Hello World"')
        self.check_enum(
            GP_ClassType,
            value='ClassType',
            child_path=[
                0,
                1],
            child_value='12')

        self.check_enum(
            GP_ProtocolType_Struct,
            value='ProtocolType',
            child_path=[0],
            child_value='12')

        self.check_enum(
            GP_ProtocolType_Class,
            value='ProtocolType',
            child_path=[
                0,
                0,
                0],
            child_summary='"Hello World"')
        self.check_enum(
            GP_ProtocolType_Class,
            value='ProtocolType',
            child_path=[
                0,
                1],
            child_value='12')

        self.check_enum(
            GP_CEnumType,
            value='CEnumType',
            child_path=[0],
            child_value='B')

        self.check_enum(
            GP_ADTEnumType,
            value='ADTEnumType',
            child_path=[
                0,
                0],
            child_value='12')

        self.check_enum(
            GP_Recursive,
            value='Recursive',
            child_path=[
                0,
                0],
            child_value='12')

