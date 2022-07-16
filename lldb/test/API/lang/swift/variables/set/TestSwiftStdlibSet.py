# TestSwiftStdlibSet.py
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
Tests that we properly vend synthetic children for Swift.Set
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftStdlibSet(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    @swiftTest
    def test_swift_stdlib_set(self):
        """Tests that we properly vend synthetic children for Swift.Set"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        self.expect(
            "frame variable",
            ordered=False,
            substrs=[
                ' = 5',
                ' = 2',
                ' = 3',
                ' = 1',
                ' = 4'])


