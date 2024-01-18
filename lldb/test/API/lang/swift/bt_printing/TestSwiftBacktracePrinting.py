# TestSwiftBacktracePrinting.py
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
Test printing Swift backtrace
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import os
import unittest2


class TestSwiftBacktracePrinting(TestBase):
    @swiftTest
    def test_swift_backtrace_printing(self):
        """Test printing Swift backtrace"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Test printing Swift backtrace"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', self.main_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        self.expect("bt", substrs=['h<T>',
                                   # FIXME: rdar://65956239 U and T are not resolved!
                                   'g<U, T>', 'pair', # '12', "Hello world",
                                   'arg1=12', 'arg2="Hello world"'])
        self.expect("breakpoint set -p other", substrs=['g<U, T>'])

