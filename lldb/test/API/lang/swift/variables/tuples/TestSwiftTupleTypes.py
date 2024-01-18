# TestSwiftTupleTypes.py
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
Test support for tuple types
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftTupleTypes(TestBase):
    @swiftTest
    def test_swift_tuple_types(self):
        """Test support for tuple types"""
        self.build()
        lldbutil.run_to_source_breakpoint(self, 'Set breakpoint here',
                                          lldb.SBFileSpec('main.swift'))

        self.expect("frame variable t --", substrs=[
            '(Int, Int, Int) t = ', '0 = 111', '1 = 222', '2 = 333'])
        self.runCmd("continue")

        self.expect("frame variable --raw-output --show-types tuple1",
                    substrs=['(length: Swift.Int, name: Swift.String) tuple1',
                             '(Swift.Int) length =',
                             'value = 123',
                             '(Swift.String) name ='])
        self.expect("frame variable --raw-output --show-types tuple2",
                    substrs=['(Swift.Int, Swift.String) tuple2',
                             '(Swift.Int) 0 =',
                             'value = 123',
                             '(Swift.String) 1 ='])
        self.expect("frame variable --raw-output --show-types tuple3",
                    substrs=['(Swift.Int, name: Swift.String) tuple3',
                             '(Swift.Int) 0 =',
                             'value = 123',
                             '(Swift.String) name ='])
        self.expect("frame variable --raw-output --show-types tuple4",
                    substrs=['(p1: ', 'Point, p2: ', '.Point) tuple4',
                             'Point) p1',
                             'FPIEEE32)', 'value = 1.25',
                             'FPIEEE32)', 'value = 2.125',
                             'Point) p2',
                             'FPIEEE32)', 'value = 4.5',
                             'FPIEEE32)', 'value = 8.75'])

