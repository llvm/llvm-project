"""
Test that we can print and call closures passed in various contexts
"""

from __future__ import print_function


import os
import time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestPassedClosures(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @expectedFailureAll(bugnumber="rdar://31816998")
    def test_static_closure_type(self):
        """This tests that we can print a closure with statically known return type."""
        self.build()
        self.static_type(False)

    @expectedFailureAll(bugnumber="rdar://31816998")
    def test_static_closure_call(self):
        """This tests that we can call a closure with statically known return type."""
        self.build()
        self.static_type(True)

    @expectedFailureAll(bugnumber="rdar://31816998")
    def test_generic_closure_type(self):
        """This tests that we can print a closure with generic return type."""
        self.build()
        self.generic_type(False)

    @expectedFailureAll(bugnumber="rdar://31816998")
    def test_generic_closure_call(self):
        """This tests that we can call a closure with generic return type."""
        self.build()
        self.generic_type(True)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def get_to_bkpt (self, bkpt_name):
        self.build()
        lldbutil.run_to_source_breakpoint(self, bkpt_name,
                                          lldb.SBFileSpec('main.swift'))

    def static_type(self, test_call):

        self.get_to_bkpt("break here for static type")
        opts = lldb.SBExpressionOptions()

        if not test_call:
            # First see that we can print the function we were passed:
            result = self.frame().EvaluateExpression("fn", opts)
            error = result.GetError()
            self.assertSuccess(error, "'fn' failed")
            self.assertTrue("() -> Swift.Int" in result.GetValue(), "Got the function name wrong: %s."%(result.GetValue()))
            self.assertTrue("() -> Swift.Int" in result.GetTypeName(), "Got the function type wrong: %s."%(result.GetTypeName()))
        
        if test_call:
            # Now see that we can call it:
            result = self.frame().EvaluateExpression("fn()", opts)
            error.result.GetError()
            self.assertSuccess(error, "'fn()' failed")
            self.assertTrue(result.GetValue() == "3", "Got the wrong value: %s"%(result.GetValue()))

    def generic_type(self, test_call):
        self.get_to_bkpt("break here for generic type")
        opts = lldb.SBExpressionOptions()

        if not test_call:
            # First see that we can print the function we were passed:
            result = self.frame().EvaluateExpression("fn", opts)
            error = result.GetError()
            self.assertSuccess(error, "'fn' failed")
            self.assertTrue("() -> A" in result.GetValue(), "Got the function name wrong: %s."%(result.GetValue()))
            self.assertTrue("() -> A" in result.GetTypeName(), "Got the function type wrong: %s."%(result.GetTypeName()))
        
        if test_call:
            # Now see that we can call it:
            result = self.frame().EvaluateExpression("fn()", opts)
            error.result.GetError()
            self.assertSuccess(error, "'fn()' failed")
            self.assertTrue(result.GetValue() == "3", "Got the wrong value: %s"%(result.GetValue()))



