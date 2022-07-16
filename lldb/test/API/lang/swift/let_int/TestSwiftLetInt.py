# TestSwiftLetInt.py
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
Test that a 'let' Int is formatted properly
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftLetIntSupport(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    @swiftTest
    def test_swift_let_int(self):
        """Test that a 'let' Int is formatted properly"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        let = self.frame().FindVariable("x")
        var = self.frame().FindVariable("y")
        lldbutil.check_variable(self, let, False, value="10")
        lldbutil.check_variable(self, var, False, value="10")

        get_arguments = False
        get_locals = True
        get_statics = False
        get_in_scope_only = True
        local_vars = self.frame().GetVariables(get_arguments, get_locals,
                                             get_statics, get_in_scope_only)
        self.assertTrue(local_vars.GetFirstValueByName("x").IsValid())
        self.assertTrue(local_vars.GetFirstValueByName("y").IsValid())
        self.assertTrue(not local_vars.GetFirstValueByName("z").IsValid())

