# TestSplitDebug.py
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
Test that split debug-info works properly
"""
import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftSplitDebug(lldbtest.TestBase):

    @swiftTest
    def test_split_debug_info(self):
        """Test split debug info"""
        self.build()
        self.do_test()

    def setUp(self):
        lldbtest.TestBase.setUp(self)

    def check_val(self, var_name, expected_val):
        value = self.frame().EvaluateExpression(var_name,
            lldb.eDynamicCanRunTarget)

        self.assertTrue(value.IsValid(),
                        "expr " + var_name + " returned a valid value")
        self.assertEquals(value.GetValue(), expected_val)

    def do_test(self):
        """Test the split debug info"""
        lldbutil.run_to_source_breakpoint(
            self, "Break here in main", lldb.SBFileSpec("main.swift"))

        self.check_val("c.c_x", "12345")
        self.check_val("c.c_y", "6789")

