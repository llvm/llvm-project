# TestSwiftTypeAliasFormatters.py
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
Test that Swift typealiases get formatted properly
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os


class TestSwiftTypeAliasFormatters(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    @swiftTest
    def test_swift_type_alias_formatters(self):
        """Test that Swift typealiases get formatted properly"""
        self.build()
        target, process, thread, a_breakpoint = \
            lldbutil.run_to_source_breakpoint(
                self, 'break here', lldb.SBFileSpec('main.swift'))

        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd(
                "settings set target.max-children-count 256",
                check=False)

        self.addTearDownHook(cleanup)

        self.expect("frame variable f", substrs=['Foo) f = (value = 12)'])
        self.expect("frame variable b", substrs=['Bar) b = (value = 24)'])

        self.runCmd('type summary add a.Foo -v -s "hello"')
        self.expect("frame variable f", substrs=['Foo) f = hello'])
        self.expect("frame variable b", substrs=['Bar) b = hello'])

        self.runCmd('type summary add a.Bar -v -s "hi"')
        self.expect("frame variable f", substrs=['Foo) f = hello'])
        self.expect("frame variable b", substrs=['Bar) b = hi'])

        self.runCmd("type summary delete a.Foo")
        self.expect("frame variable f", substrs=['Foo) f = (value = 12)'])
        self.expect("frame variable b", substrs=['Bar) b = hi'])

        self.runCmd("type summary delete a.Bar")
        self.runCmd("type summary add -C no -v a.Foo -s hello")
        self.expect("frame variable f", substrs=['Foo) f = hello'])
        self.expect("frame variable b", substrs=['Bar) b = (value = 24)'])
