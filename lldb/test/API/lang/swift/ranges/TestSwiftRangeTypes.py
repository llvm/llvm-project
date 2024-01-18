# TestSwiftRangeTypes.py
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
Test the Swift.Range<T> type
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftRangeType(TestBase):
    @swiftTest
    def test_swift_range_type(self):
        """Test the Swift.Range<T> type"""
        self.build()
        lldbutil.run_to_source_breakpoint(
                self, 'Set breakpoint here', lldb.SBFileSpec("main.swift"))
        self.expect("frame variable a", substrs=[
                    '(ClosedRange<Int>) a = 1...100'])
        self.expect("frame variable b", substrs=['(Range<Int>) b = 1..<100'])
        self.expect("frame variable c", substrs=[
                    '(ClosedRange<Int>) c = 1...100'])
        self.expect("frame variable d", substrs=[
                    '(Range<Int>) d = 1..<100'])
