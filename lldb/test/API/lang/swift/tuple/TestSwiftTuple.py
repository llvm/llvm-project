# TestSwiftTuple.py
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
Test that LLDB understands tuple lowering
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftTuple(TestBase):
    @swiftTest
    def test_swift_tuples(self):
        """Test that LLDB understands tuple lowering"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        self.expect("frame variable s", substrs=['0 = 123', '1 = 0x'])

        self.expect("expression s.tup.0", substrs=['123'])
        self.expect("expression s.tup.1()", substrs=['321'])
