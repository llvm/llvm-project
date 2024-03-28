# TestSwiftAtObjCIvars.py
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
Check that we correctly find offsets for ivars of Swift @objc types
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftAtObjCIvars(TestBase):
    def check_foo(self, theFoo):
        x = theFoo.GetChildMemberWithName("x")
        y = theFoo.GetChildMemberWithName("y")
        lldbutil.check_variable(self, x, False, value='12')
        lldbutil.check_variable(self, y, False, '"12"')

    @skipUnlessDarwin
    @swiftTest
    def test_swift_at_objc_ivars(self):
        """Test ObjC instance variables"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set a breakpoint here', lldb.SBFileSpec('main.swift'))

        foo_objc = self.frame().FindVariable("foo_objc")
        foo_swift = self.frame().FindVariable("foo_swift")

        self.check_foo(foo_objc)
        self.check_foo(foo_swift)
