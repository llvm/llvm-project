# TestSwiftObjCOptionals.py
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
Check formatting for T? and T! when T is an ObjC type
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftObjCOptionalType(TestBase):

    @swiftTest
    @skipUnlessDarwin
    def test_swift_objc_optional_type(self):
        """Check formatting for T? and T! when T is an ObjC type"""
        self.build()
        self.do_check_consistency()
        self.do_check_visuals()
        self.do_check_api()

    def setUp(self):
        TestBase.setUp(self)

    def do_check_consistency(self):
        """Check formatting for T? and T! when T is an ObjC type"""
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

    def do_check_visuals(self):
        """Check formatting for T? and T! when T is an ObjC type"""
        self.expect(
            "frame variable optColor_Some",
            substrs=['Color?) optColor_Some = 0x'])
        self.expect(
            "frame variable uoptColor_Some",
            substrs=['Color?) uoptColor_Some = 0x'])

        self.expect("frame variable optColor_None", substrs=['nil'])
        self.expect("frame variable uoptColor_None", substrs=['nil'])

    def do_check_api(self):
        """Check formatting for T? and T! when T is an ObjC type"""
        optColor_Some = self.frame().FindVariable("optColor_Some")

        # SwiftOptionalSyntheticFrontEnd passes GetNumChildren
        # through to the .some object.  NSColor has no children.
        lldbutil.check_variable(
            self,
            optColor_Some,
            use_dynamic=False,
            num_children=0)
        uoptColor_Some = self.frame().FindVariable("uoptColor_Some")
        lldbutil.check_variable(
            self,
            uoptColor_Some,
            use_dynamic=False,
            num_children=0)

