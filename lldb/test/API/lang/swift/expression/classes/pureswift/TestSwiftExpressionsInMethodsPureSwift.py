# TestSwiftExpressionsInMethodsPureSwift.py
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


class TestExpressionsInSwiftMethodsPureSwift(TestBase):
    @swiftTest
    def test_expressions_in_methods(self):
        """Tests that we can run simple Swift expressions correctly"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Stop here in Pure Swift class', lldb.SBFileSpec('main.swift'))

        lldbutil.check_expression(self, self.frame(), "m_computed_ivar == 5", "true")
        lldbutil.check_expression(self, self.frame(), "m_ivar", "10", use_summary=False)
        lldbutil.check_expression(self, self.frame(), "self.m_ivar == 11", "false")
