# TestSwiftMultipayloadEnum.py
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
Test that LLDB understands generic enums with more than one payload type
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftMultipayloadEnum(TestBase):
    @swiftTest
    def test_swift_multipayload_enum(self):
        """Test that LLDB understands generic enums with more than one payload type"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        self.expect("frame variable one", substrs=['One', '1234'])
        self.expect("frame variable two", substrs=['TheOther', '"some value"'])

        self.expect("expression one", substrs=['One', '1234'])
        self.expect("expression two", substrs=['TheOther', '"some value"'])
