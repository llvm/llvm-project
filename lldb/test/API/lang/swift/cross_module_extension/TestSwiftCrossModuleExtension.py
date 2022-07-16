# TestSwiftCrossModuleExtension.py
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
Test that we correctly find private extension decls across modules
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import unittest2

class TestSwiftCrossModuleExtension(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    @skipUnlessDarwin
    @swiftTest
    def test_cross_module_extension(self):
        """Test that we correctly find private extension decls across modules"""
        self.build()
        target, process, thread, a_breakpoint = \
            lldbutil.run_to_source_breakpoint(
                self, 'break here', lldb.SBFileSpec('moda.swift'),
                exe_name = self.getBuildArtifact("main"))
        b_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec('modb.swift'))
        self.assertTrue(b_breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)
        frame = thread.frames[0]
        self.assertTrue(frame, "Frame 0 is valid.")

        var = frame.FindVariable("a")
        child_v = var.GetChildMemberWithName("v")
        lldbutil.check_variable(self, var, False, typename="moda.S.A")
        lldbutil.check_variable(self, child_v, False, value="1")

        process.Continue()
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, b_breakpoint)

        self.assertTrue(len(threads) == 1)
        frame = threads[0].frames[0]
        self.assertTrue(frame, "Frame 0 is valid.")

        var = frame.FindVariable("a")
        child_v = var.GetChildMemberWithName("v")
        lldbutil.check_variable(self, var, False, typename="moda.S.A")
        lldbutil.check_variable(self, child_v, False, value="3")

