# TestSwiftStructInit.py
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
Test that we display self correctly for an inline-initialized struct
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftStructInit(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    @swiftTest
    @skipIf(oslist=['windows'])
    def test_swift_struct_init(self):
        """Test that we display self correctly for an inline-initialized struct"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        theself = self.frame().FindVariable("self")
        var_a = theself.GetChildMemberWithName("a")
        lldbutil.check_variable(self, var_a, False, value="12")

        self.runCmd("next")

        var_b = theself.GetChildMemberWithName("b")
        lldbutil.check_variable(self, var_b, False, '"Hey"')
