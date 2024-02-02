# TestSwiftBool.py
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
Test that we can inspect Swift Bools - they are 8 bit entities with only the
LSB significant.  Make sure that works.
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftBool(TestBase):
    @swiftTest
    def test_swift_bool(self):
        """Test that we can inspect various Swift bools"""
        self.build()
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.assertGreater(thread.GetNumFrames(), 0)
        frame = thread.GetSelectedFrame()
        
        true_vars = ["reg_true", "odd_true", "odd_true_works", "odd_false_works"]
        for name in true_vars:
            var = frame.FindVariable(name)
            summary = var.GetSummary()
            self.assertTrue(summary == "true", "%s should be true, was: %s"%(name, summary))

        false_vars = ["reg_false", "odd_false"]
        for name in false_vars:
            var = frame.FindVariable(name)
            summary = var.GetSummary()
            self.assertTrue(summary == "false", "%s should be false, was: %s"%(name, summary))


