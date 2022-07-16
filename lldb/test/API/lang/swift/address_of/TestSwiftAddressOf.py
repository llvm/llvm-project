# TestSwiftAddressOf.py
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
Test that AddressOf returns sensible results
"""
import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftAddressOf(lldbtest.TestBase):

    def setUp(self):
        lldbtest.TestBase.setUp(self)

    def check_variable(self, name, is_reference, contents = 0):
        var = self.frame().FindVariable(name)
        self.assertTrue(var.IsValid(), "Couldn't find %s var: %s"%(name, var.GetError().GetCString()))
        if is_reference:
            self.assertTrue(var.GetType().IsReferenceType(), name + "was not supposed to be a reference.")
            addr_value = var.GetValueAsUnsigned()
        else:
            self.assertFalse(var.GetType().IsReferenceType(), name + "was supposed to be a reference.")
            addr_val = var.AddressOf()
            self.assertSuccess(addr_val.GetError(), "AddressOf didn't return a good variable")
            addr_value = addr_val.GetValueAsUnsigned()
            
        # FIXME: I want to use SBTarget::CreateValueFromAddress to make the
        # same variable from this address & type, and then compare the ivars.
        # But SBTarget::CreateValueFromAddress isn't hooked up for Swift, so
        # for now I'm just telling myself that I could read the memory.
        error = lldb.SBError()
        bytes = self.process.ReadMemory(addr_value, var.GetByteSize(), error)
        self.assertSuccess(error)

        
    @swiftTest
    def test_any_type(self):
        """Test the Any type"""
        self.build()

        # Create the target
        target = self.dbg.CreateTarget(self.getBuildArtifact())
        self.assertTrue(target, lldbtest.VALID_TARGET)

        # Set the breakpoints:
        bkpt_list = ["main", "takes_class", "takes_struct", "takes_inout"]
        breakpoints = {}
        for bkpt_text in bkpt_list:
            breakpoints[bkpt_text] = target.BreakpointCreateBySourceRegex(
                'Break here in ' + bkpt_text, lldb.SBFileSpec('main.swift'))
        self.assertTrue(
            breakpoints[bkpt_text].GetNumLocations() > 0,
            lldbtest.VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        self.process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(self.process, lldbtest.PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            self.process, breakpoints["main"])

        self.assertTrue(len(threads) == 1)

        self.check_variable("c", True, 12345)
        self.check_variable("s", False, 12345)
        self.check_variable("int", False, 100)

        lldbutil.continue_to_breakpoint(self.process, breakpoints["takes_class"])
        self.check_variable("in_class", True, 12345)
        
        lldbutil.continue_to_breakpoint(self.process, breakpoints["takes_struct"])
        self.check_variable("in_struct", False, 12345)
        
        lldbutil.continue_to_breakpoint(self.process, breakpoints["takes_inout"])

        # Inout sugar is currently not preserved by the compiler so
        # the inout type appears as direct.
        self.check_variable("in_struct", False, 12345)
        
