"""
Test expression operations in class constrained protocols
"""

from __future__ import print_function


import os
import time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class TestClassConstrainedProtocol(TestBase):
    @swiftTest
    def test_extension_weak_self(self):
        """Test that we can reconstruct weak self captured in a class constrained protocol."""
        self.build()
        self.do_self_test("Break here for weak self", needs_dynamic=False)

    @swiftTest
    def test_extension_self (self):
        """Test that we can reconstruct self in method of a class constrained protocol."""
        self.build()
        self.do_self_test("Break here in class protocol", needs_dynamic=False)

    @swiftTest
    def test_method_weak_self(self):
        """Test that we can reconstruct weak self capture in method of a class conforming to a class constrained protocol."""
        self.build()
        self.do_self_test("Break here for method weak self", needs_dynamic=False)

    @swiftTest
    def test_method_self(self):
        """Test that we can reconstruct self in method of a class conforming to a class constrained protocol."""
        self.build()
        self.do_self_test("Break here in method", needs_dynamic=False)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def check_self(self, bkpt_pattern, needs_dynamic):
        opts = lldb.SBExpressionOptions()
        if needs_dynamic:
            opts.SetFetchDynamicValue(lldb.eNoDynamicValues)
            result = self.frame().EvaluateExpression("self", opts)
            error = result.GetError()
            self.assertTrue("self" in error.GetCString())
            self.assertTrue("run-target" in error.GetCString())
        opts.SetFetchDynamicValue(lldb.eDynamicCanRunTarget)
        result = self.frame().EvaluateExpression("self", opts)
        error = result.GetError()
        self.assertSuccess(error,
                           "'self' expression failed at '%s'" % bkpt_pattern)
        f_ivar = result.GetChildMemberWithName("f")
        self.assertTrue(f_ivar.IsValid(),
                        "Could not find 'f' in self at '%s'"%(bkpt_pattern))
        self.assertTrue(f_ivar.GetValueAsSigned() == 12345,
                        "Wrong value for f: %d"%(f_ivar.GetValueAsSigned()))

    def do_self_test(self, bkpt_pattern, needs_dynamic):
        lldbutil.run_to_source_breakpoint(
            self, bkpt_pattern, lldb.SBFileSpec('main.swift'))
        self.check_self(bkpt_pattern, needs_dynamic)
