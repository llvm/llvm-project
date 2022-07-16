# TestSwiftBreakByPartialName.py
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
Tests that we can break on a partial name of a Swift function
Effectively tests our chopper of Swift demangled names
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class SwiftPartialBreakTest(TestBase):

    @swiftTest
    def test_swift_partial_break(self):
        """Tests that we can break on a partial name of a Swift function"""
        self.build()
        self.break_commands()

    def setUp(self):
        TestBase.setUp(self)

    def break_commands(self):
        """Tests that we can break on a partial name of a Swift function"""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_symbol(self, "incr")
        lldbutil.run_break_set_by_symbol(self, "Accumulator.decr")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        self.expect("frame select 0", substrs=['Accumulator', 'incr'])

        self.runCmd("continue", RUN_SUCCEEDED)

        self.expect("frame select 0", substrs=['Accumulator', 'decr'])

        self.runCmd("continue", RUN_SUCCEEDED)

        self.expect("frame select 0", substrs=['Accumulator', 'incr'])
