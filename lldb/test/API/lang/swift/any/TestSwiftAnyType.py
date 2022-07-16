# TestSwiftAnyType.py
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
Test the Any type
"""
import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftAnyType(lldbtest.TestBase):

    @swiftTest
    def test_any_type(self):
        """Test the Any type"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        frame = thread.frames[0]
        var_c = frame.FindVariable("c")
        var_c_x = var_c.GetChildMemberWithName("x")
        var_q = frame.FindVariable("q")
        lldbutil.check_variable(self, var_c_x, True, value="12")
        lldbutil.check_variable(self, var_q, True, value="12")

        self.expect("expression -d run -- q", substrs=['12'])
        self.expect("frame variable -d run -- q", substrs=['12'])
