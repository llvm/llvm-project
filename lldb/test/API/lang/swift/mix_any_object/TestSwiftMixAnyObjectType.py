# TestSwiftMixAnyObjectType.py
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
Test the AnyObject type in different combinations
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftMixAnyObjectType(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test_any_object_type(self):
        """Test the AnyObject type in different combinations"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, '// break here', lldb.SBFileSpec('main.swift'))

        self.expect(
            'frame variable -d run -- cls',
            substrs=['text = "Instance of MyClass"'])
        self.expect(
            'expr -d run -- cls',
            substrs=['text = "Instance of MyClass"'])

        self.expect(
            'frame variable -d run -- any',
            substrs=['text = "Instance of MyClass"'])
        self.expect(
            'expr -d run -- any',
            substrs=['text = "Instance of MyClass"'])

        self.expect(
            'frame variable -d run -- opt',
            substrs=['text = "Instance of MyClass"'])
        self.expect(
            'expr -d run -- opt',
            substrs=['text = "Instance of MyClass"'])

        self.expect(
            'frame variable -d run -- dict',
            ordered=False,
            substrs=[
                'key = "One"',
                'text = "Instance One"',
                'key = "Three"',
                'text = "Instance of MyClass"',
                'key = "Two"',
                'text = "Instance Two"'])
        self.expect(
            'expr -d run -- dict',
            ordered=False,
            substrs=[
                'key = "One"',
                'text = "Instance One"',
                'key = "Three"',
                'text = "Instance of MyClass"',
                'key = "Two"',
                'text = "Instance Two"'])
