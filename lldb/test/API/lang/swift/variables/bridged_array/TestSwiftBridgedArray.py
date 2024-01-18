# TestSwiftBridgedArray.py
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
Check formatting for Swift.Array<T> that are bridged from ObjC
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftBridgedArray(TestBase):
    @skipUnlessDarwin
    @swiftTest
    @expectedFailureAll(bugnumber="<rdar://problem/32024572>")
    def test_swift_bridged_array(self):
        """Check formatting for Swift.Array<T> that are bridged from ObjC"""
        self.build()
        lldbutil.run_to_source_breakpoint(self, 'break here',
                                          lldb.SBFileSpec('main.swift'))

        self.expect(
            "frame variable -d run -- swarr",
            substrs=['Int(123456)', 'Int32(234567)', 'UInt16(45678)', 'Double(1.250000)', 'Float(2.500000)'])
        self.expect(
            "expression -d run -- swarr",
            substrs=['Int(123456)', 'Int32(234567)', 'UInt16(45678)', 'Double(1.250000)', 'Float(2.500000)'])

