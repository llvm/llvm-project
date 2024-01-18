# TestSwiftAnyObjectType.py
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
Test the AnyObject type
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftAnyObjectType(TestBase):
    @swiftTest
    def test_any_object_type(self):
        """Test the AnyObject type"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        frame = thread.frames[0]

        var_object = frame.FindVariable("object", lldb.eNoDynamicValues)

        lldbutil.check_variable(
            self,
            var_object,
            use_dynamic=False,
            typename="Swift.AnyObject")
        lldbutil.check_variable(
            self,
            var_object,
            use_dynamic=True,
            typename="a.SomeClass")
        var_object_x = var_object.GetDynamicValue(
            lldb.eDynamicCanRunTarget).GetChildMemberWithName("x")
        lldbutil.check_variable(
            self,
            var_object_x,
            use_dynamic=False,
            value='12',
            typename="Swift.Int")
