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
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Tests that we can break and display simple types"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here', self.main_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]

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
