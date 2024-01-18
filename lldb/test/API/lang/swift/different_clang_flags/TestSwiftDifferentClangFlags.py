# TestSwiftDifferentClangFlags.py
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
Test that we use the right compiler flags when debugging
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import unittest2

import sys
if sys.version_info.major == 2:
    import commands as subprocess
else:
    import subprocess

def execute_command(command):
    # print '%% %s' % (command)
    (exit_status, output) = subprocess.getstatusoutput(command)
    # if output:
    #     print output
    # print 'status = %u' % (exit_status)
    return exit_status


class TestSwiftDifferentClangFlags(TestBase):
    @skipUnlessDarwin
    @swiftTest
    @skipIf(
        debug_info=decorators.no_match("dsym"),
        bugnumber="This test requires a stripped binary and a dSYM")
    def test_swift_different_clang_flags(self):
        """Test that we use the right compiler flags when debugging"""
        self.build()
        target, process, thread, modb_breakpoint = \
            lldbutil.run_to_source_breakpoint(
                self, 'break here', lldb.SBFileSpec("modb.swift"),
                exe_name=self.getBuildArtifact("main"),
                extra_images=['moda', 'modb'])

        main_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here',lldb.SBFileSpec('main.swift'))
        self.assertTrue(
            modb_breakpoint.GetNumLocations() > 0,
            VALID_BREAKPOINT)

        var = self.frame().FindVariable("myThree")
        three = var.GetChildMemberWithName("three")
        lldbutil.check_variable(self, var, False, typename="modb.MyStruct")
        lldbutil.check_variable(self, three, False, value="3")

        process.Continue()
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, main_breakpoint)

        var = self.frame().FindVariable("a")
        lldbutil.check_variable(self, var, False, value="2")
        var = self.frame().FindVariable("b")
        lldbutil.check_variable(self, var, False, value="3")

        var = self.frame().EvaluateExpression("fA()")
        lldbutil.check_variable(self, var, False, value="2")

