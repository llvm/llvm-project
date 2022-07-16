# TestResilientObjectInOptional.py
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
Test that we can extract a resilient object from an Optional
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2

class TestResilientObjectInOptional(TestBase):

    @skipUnlessDarwin
    @swiftTest
    def test_optional_of_resilient(self):
        """Test that can extract resilient objects from an Optional"""
        self.build()
        self.doTest()

    def setUp(self):
        TestBase.setUp(self)

    def doTest(self):
        target, process, thread, breakpoint = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"),
            extra_images=['mod'])

        frame = thread.frames[0]
        
        # First try getting a non-resilient optional, to make sure that
        # part isn't broken:
        t_opt_var = frame.FindVariable("t_opt")
        self.assertSuccess(t_opt_var.GetError(), "Made t_opt value object")
        t_a_var = t_opt_var.GetChildMemberWithName("a")
        self.assertSuccess(t_a_var.GetError(), "The child was a")
        lldbutil.check_variable(self, t_a_var, False, value="2")
        
        # Make sure we can print an optional of a resilient type...
        # If we got the value out of the optional correctly, then
        # it's child will be "a".
        # First do this with "frame var":
        opt_var = frame.FindVariable("s_opt")
        self.assertSuccess(opt_var.GetError(), "Made s_opt value object")
        a_var = opt_var.GetChildMemberWithName("a")
        self.assertSuccess(a_var.GetError(), "The resilient child was 'a'")
        lldbutil.check_variable(self, a_var, False, value="1")
