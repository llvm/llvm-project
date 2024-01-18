# TestCGTypes.py
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
Test that we are able to properly format basic CG types
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftCoreGraphicsTypes(TestBase):
    @swiftTest
    @skipUnlessDarwin
    def test_swift_coregraphics_types(self):
        """Test that we are able to properly format basic CG types"""
        self.build()
        self.do_test()

    def do_test(self):
        """Test that we are able to properly format basic CG types"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect('frame variable f', substrs=[' f = 1'])
        self.expect('frame variable p', substrs=[' p = (x = 1, y = 1)'])
        self.expect('frame variable r', substrs=[
            ' r = (origin = (x = 0, y = 0), size = (width = 0, height = 0))'])

        self.expect('expr f', substrs=[' = 1'])
        self.expect('expr p', substrs=[' = (x = 1, y = 1)'])
        self.expect(
            'expr r',
            substrs=[' = (origin = (x = 0, y = 0), size = (width = 0, height = 0))'])

        self.expect('po f', substrs=['1.0'])
        self.expect('po p', substrs=['x : 1.0', 'y : 1.0'])
        self.expect(
            'po r',
            substrs=[
                'x : 0.0',
                'y : 0.0',
                'width : 0.0',
                'height : 0.0'])


