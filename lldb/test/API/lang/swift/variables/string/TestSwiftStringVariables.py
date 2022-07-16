# coding=utf-8

# TestSwiftStringVariables.py
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
import unittest2


class TestSwiftStringVariables(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    @swiftTest
    def test_swift_string_variables(self):
        """Test that Swift.String formats properly"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        s1 = self.frame().FindVariable("s1")
        s2 = self.frame().FindVariable("s2")

        lldbutil.check_variable(self, s1, summary='"Hello world"')
        lldbutil.check_variable(self, s2, summary='"ΞΕΛΛΘ"')

        TheVeryLongOne = self.frame().FindVariable("TheVeryLongOne")
        summaryOptions = lldb.SBTypeSummaryOptions()
        summaryOptions.SetCapping(lldb.eTypeSummaryUncapped)
        uncappedSummaryStream = lldb.SBStream()
        TheVeryLongOne.GetSummary(uncappedSummaryStream, summaryOptions)
        uncappedSummary = uncappedSummaryStream.GetData()
        self.assertTrue(uncappedSummary.find("someText") > 0,
                        "uncappedSummary does not include the full string")
        summaryOptions.SetCapping(lldb.eTypeSummaryCapped)
        cappedSummaryStream = lldb.SBStream()
        TheVeryLongOne.GetSummary(cappedSummaryStream, summaryOptions)
        cappedSummary = cappedSummaryStream.GetData()
        self.assertTrue(
            cappedSummary.find("someText") <= 0,
            "cappedSummary includes the full string")
        self.assertTrue(
            cappedSummary.endswith('"...'),
            "cappedSummary ends with quote dot dot dot")

        IContainZerosASCII = self.frame().FindVariable("IContainZerosASCII")
        IContainZerosUnicode = self.frame().FindVariable("IContainZerosUnicode")
        IContainEscapes = self.frame().FindVariable("IContainEscapes")

        lldbutil.check_variable(
            self,
            IContainZerosASCII,
            summary='"a\\0b\\0c\\0d"')
        lldbutil.check_variable(
            self,
            IContainZerosUnicode,
            summary='"HFIHЗIHF\\0VЭHVHЗ90HGЭ"')
        lldbutil.check_variable(
            self,
            IContainEscapes,
            summary='"Hello\\u{8}\\n\\u{8}\\u{8}\\nGoodbye"')

        self.expect(
            'expression -l objc++ -- (char*)"Hello\b\b\b\b\bGoodbye"',
            substrs=['"Hello\\b\\b\\b\\b\\bGoodbye"'])

