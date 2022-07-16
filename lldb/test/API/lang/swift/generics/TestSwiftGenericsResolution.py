# TestSwiftGenericsResolution.py
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
Check that we can correctly figure out the dynamic type of generic things
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class SwiftDynamicTypeGenericsTest(TestBase):

    @swiftTest
    def test_genericresolution_commands(self):
        """Check that we can correctly figure out the dynamic type of generic things"""
        self.build()
        self.genericresolution_commands()

    def setUp(self):
        TestBase.setUp(self)

    def genericresolution_commands(self):
        """Check that we can correctly figure out the dynamic type of generic things"""
        lldbutil.run_to_source_breakpoint(self, "//Break here", lldb.SBFileSpec("main.swift"))

        self.expect(
            "frame variable -d run",
            substrs=[
                "(Int) x = 123",
                "(a.OtherClass<Int>) self = 0x",
                "a.AClass<Swift.Int> = {}",
                "v = 1234567"])
        self.runCmd("continue")
        self.expect(
            "frame variable -d run",
            substrs=[
                '(String) x = "hello world again"',
                '(Int) v = 1'])
        self.runCmd("continue")
        self.expect(
            "frame variable -d run",
            substrs=[
                '(a.Pair<a.Generic<Int>, a.Pair<String, a.Generic<String>>>) self = 0x',
                'one = ',
                'v = 193627',
                'two = 0x',
                'one = "hello"',
                'two = (v = "world")'])
        self.runCmd("continue")
        self.expect(
            "frame variable -d run",
            substrs=[
                '(a.Pair<a.Generic<Double>, a.Generic<a.Pair<String, String>>>) self = 0',
                'one = ',
                'v = 3.1',
                'two = {',
                'v = 0x',
                '(one = "this is", two = "a good thing")'])
        self.runCmd("continue")
        self.expect(
            "frame variable -d run",
            substrs=[
                "(Int) x = 5",
                '(String) y = "hello world"',
                "(a.OtherClass<Int>) self = 0x",
                "a.AClass<Swift.Int> = {}",
                "v = 1234567"])
        self.runCmd("continue")
        self.expect(
            "frame variable -d run",
            substrs=[
                "(a.Outer<Int>.Inner) self ="
            ])
