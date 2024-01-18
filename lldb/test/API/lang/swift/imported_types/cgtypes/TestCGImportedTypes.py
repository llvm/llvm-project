# TestCGImportedTypes.py
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
Test that we are able to deal with C-imported types (from CoreGraphics)
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftCGImportedTypes(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test_swift_cg_imported_types(self):
        """Test that we are able to deal with C-imported types from CoreGraphics"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        rect = self.frame().FindVariable("cgrect")
        self.assertTrue(rect.IsValid(), "Got the cgrect variable")
        origin_var = rect.GetChildMemberWithName("origin")
        self.assertTrue(origin_var.IsValid(), "Got origin from cgrect")
        x_var = origin_var.GetChildMemberWithName("x")
        self.assertTrue(x_var.IsValid(), "Got valid x from cgrect.origin")
        x_native = x_var.GetChildMemberWithName("native")
        self.assertTrue(
            x_native.IsValid(),
            "Got valid native from cgrect.origin.x")
        self.assertEquals(x_native.GetValue(), "10", "Value of x is correct")
