# TestSwiftDynamicValue.py
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
Tests that dynamic values work correctly for Swift
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class SwiftDynamicValueTest(TestBase):

    @swiftTest
    def test_dynamic_value(self):
        """Tests that dynamic values work correctly for Swift"""
        self.build()
        self.dynamic_val_commands()

    def setUp(self):
        TestBase.setUp(self)

    def dynamic_val_commands(self):
        """Tests that dynamic values work correctly for Swift"""
        lldbutil.run_to_source_breakpoint(self, "// Set a breakpoint here", lldb.SBFileSpec("main.swift"))

        self.expect(
            "frame variable",
            substrs=[
                "AWrapperClass) aWrapper",
                "SomeClass) anItem = ",
                "x = ",
                "Base<Int>) aBase = 0x",
                "v = 449493530"])
        self.expect(
            "frame variable -d run --show-types",
            substrs=[
                "AWrapperClass) aWrapper",
                "YetAnotherClass) anItem = ",
                "x = ",
                "y = ",
                "z = ",
                "Derived<Int>) aBase = 0x",
                "Base<Int>)",
                ".Base<Swift.Int> = {",
                "v = 449493530",
                "q = 3735928559"])
        self.runCmd("continue")
        self.expect(
            "frame variable",
            substrs=[
                "AWrapperClass) aWrapper",
                "SomeClass) anItem = ",
                "x = ",
                "Base<Int>) aBase = 0x",
                "v = 449493530"])
        self.expect(
            "frame variable -d run --show-types",
            substrs=[
                "AWrapperClass) aWrapper",
                "YetAnotherClass) anItem = ",
                "x = ",
                "y = ",
                "z = ",
                "Derived<Int>) aBase = 0x",
                "Base<Int>)",
                ".Base<Swift.Int> = {",
                "v = 449493530",
                "q = 3735928559"])
