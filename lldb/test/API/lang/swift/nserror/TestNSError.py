# TestNSError.py
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
Tests that Swift displays NSError correctly
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class SwiftNSErrorTest(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test_swift_nserror(self):
        """Tests that Swift displays NSError correctly"""
        self.build()
        self.nserror_commands(check_userInfo=False)

    @skipUnlessDarwin
    @swiftTest
    @skipIf(bugnumber="rdar://71549869")
    def test_swift_nserror_fails(self):
        """Tests that Swift displays NSError correctly"""
        self.build()
        self.nserror_commands(check_userInfo=True)

    def nserror_commands(self, check_userInfo=True):
        """Tests that Swift displays NSError correctly"""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_source_regexp(
            self, "// Set a breakpoint here")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs=['stopped', 'stop reason = breakpoint'])

        substrs = [
            '0 = " "',
            '0 = "x+y"',
            '1 = 0x',
            'domain: "lldbrocks" - code: 3133079277 {',
            'domain: "lldbrocks" - code: 0 {'
        ]

        if check_userInfo:
            substrs.extend([
                '_userInfo = 2 key/value pairs {',
                    '[0] = {',
                    '[1] = {',
                        'key = "x"',
                        'key = "y"',
                        'value = 0',
                        'value = 3',
                        'value = 4'])

        self.expect(
            "frame variable -d run --ptr-depth=2",
            ordered=False,
            substrs=substrs)
