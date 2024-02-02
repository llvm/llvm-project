# TestFunctionVariables.py
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
Tests that Enum variables display correctly
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestFunctionVariables(TestBase):
    @swiftTest
    def test_function_variables(self):
        """Tests that function type variables display correctly"""
        self.build()
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.assertGreater(thread.GetNumFrames(), 0)
        frame = thread.GetSelectedFrame()

        # Get the function pointer variable from our frame
        func_ptr_value = frame.FindVariable('func_ptr')

        # Grab the function pointer value as an unsigned load address
        func_ptr_addr = func_ptr_value.GetValueAsUnsigned()

        # Resolve the load address into a section + offset address
        # (lldb.SBAddress)
        func_ptr_so_addr = target.ResolveLoadAddress(func_ptr_addr)

        # Get the debug info function for this address
        func_ptr_function = func_ptr_so_addr.GetFunction()

        # Make sure the function pointer correctly resolved to our a.bar
        # function
        self.assertEqual('a.bar() -> ()', func_ptr_function.name)

