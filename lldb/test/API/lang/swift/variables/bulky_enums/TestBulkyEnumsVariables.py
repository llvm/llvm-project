# TestBulkyEnumsVariables.py
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
Tests that large-size Enum variables display correctly
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestBulkyEnumVariables(TestBase):

    @swiftTest
    def test_bulky_enum_variables(self):
        """Tests that large-size Enum variables display correctly"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)

    def do_test(self):
        """Tests that large-size Enum variables display correctly"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        self.expect(
            'frame variable e',
            substrs=[
                'e = X ',
                '0 = "hello world"',
                'b = (a = 100, b = 200)',
                'a = (a = 300, b = 400)'])

