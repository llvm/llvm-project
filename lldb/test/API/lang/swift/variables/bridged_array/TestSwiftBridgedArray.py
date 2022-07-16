# TestSwiftBridgedArray.py
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
Check formatting for Swift.Array<T> that are bridged from ObjC
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftBridgedArray(TestBase):

    @skipUnlessDarwin
    @swiftTest
    @expectedFailureAll(bugnumber="<rdar://problem/32024572>")
    def test_swift_bridged_array(self):
        """Check formatting for Swift.Array<T> that are bridged from ObjC"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Check formatting for Swift.Array<T> that are bridged from ObjC"""
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

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]

        self.expect(
            "frame variable -d run -- swarr",
            substrs=['Int(123456)', 'Int32(234567)', 'UInt16(45678)', 'Double(1.250000)', 'Float(2.500000)'])
        self.expect(
            "expression -d run -- swarr",
            substrs=['Int(123456)', 'Int32(234567)', 'UInt16(45678)', 'Double(1.250000)', 'Float(2.500000)'])

