# TestSwiftClassTypes.py
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
Test swift Class types
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftClassTypes(TestBase):
    @swiftTest
    def test_swift_class_types(self):
        """Test swift Class types"""
        self.build()
        lldbutil.run_to_source_breakpoint(self, 'Set breakpoint here',
                                          lldb.SBFileSpec('main.swift'))
        self.expect("frame variable --show-types f",
                    substrs=['Foo) f = 0x',
                             'Base) ', '.Base = {',
                             '(String) b = ',
                             '(Int) x = 12',
                             '(Float) y = 2.25'])

