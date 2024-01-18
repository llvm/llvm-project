# TestSwiftGenericTypes.py
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
Test support for generic types
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftGenericTypes(TestBase):
    @swiftTest
    def test_swift_generic_types(self):
        """Test support for generic types"""
        self.build()
        lldbutil.run_to_source_breakpoint(self, 'Set breakpoint here',
                                          lldb.SBFileSpec('main.swift'))

        self.expect("frame variable -d no-dynamic-values object",
                    substrs=['(JustSomeType) object = 0x'])
        self.expect(
            "frame variable -d run-target -- object",
            substrs=['(Int) object = 255'])

        self.runCmd("continue")
        self.runCmd("frame select 0")

        self.expect("frame variable --show-types c",
                    substrs=['(Int) c = 255'])

        self.expect("frame variable --raw-output --show-types o_some",
                    substrs=['(Swift.Optional<Swift.String>) o_some = some {',
                             '(Swift.String) some ='])
        self.expect("frame variable --raw-output --show-types o_none",
                    substrs=['(Swift.Optional<Swift.String>) o_none = none'])

        self.expect(
            "frame variable o_some o_none",
            substrs=[
                '(String?) o_some = "Hello"',
                '(String?) o_none = nil'])
