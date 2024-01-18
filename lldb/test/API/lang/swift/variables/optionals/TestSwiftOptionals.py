# TestSwiftOptionals.py
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
Check formatting for T? and T!
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftOptionalType(TestBase):
    @swiftTest
    def test_swift_optional_type(self):
        """Check formatting for T? and T!"""
        self.do_check_consistency()
        self.do_check_visuals()
        self.do_check_api()

    def do_check_consistency(self):
        """Check formatting for T? and T!"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

    def do_check_visuals(self):
        """Check formatting for T? and T!"""
        self.expect(
            "frame variable optS_Some",
            substrs=[
                'a = 12',
                'b = "Hello world"'])
        self.expect(
            "frame variable uoptS_Some",
            substrs=[
                'a = 12',
                'b = "Hello world"'])

        self.expect("frame variable optString_Some", substrs=['"hello"'])
        self.expect("frame variable uoptString_Some", substrs=['"hello"'])

        self.expect("frame variable optS_None", substrs=['nil'])
        self.expect("frame variable uoptS_None", substrs=['nil'])

        self.expect("frame variable optString_None", substrs=['nil'])
        self.expect("frame variable uoptString_None", substrs=['nil'])

        self.expect("frame variable optTrue", substrs=['Bool?', 'true'])
        self.expect("frame variable optFalse", substrs=['Bool?','false'])
        self.expect("frame variable optNil", substrs=['Bool?', 'nil'])

    def do_check_api(self):
        """Check formatting for T? and T!"""
        optS_Some = self.frame().FindVariable("optS_Some")
        lldbutil.check_variable(
            self,
            optS_Some,
            use_dynamic=False,
            num_children=2)
        uoptS_Some = self.frame().FindVariable("uoptS_Some")
        lldbutil.check_variable(
            self,
            uoptS_Some,
            use_dynamic=False,
            num_children=2)

        optString_None = self.frame().FindVariable("optString_None")
        lldbutil.check_variable(
            self,
            optString_None,
            use_dynamic=False,
            num_children=0)
        uoptString_None = self.frame().FindVariable("uoptString_None")
        lldbutil.check_variable(
            self,
            uoptString_None,
            use_dynamic=False,
            num_children=0)

        optString_Some = self.frame().FindVariable("optString_Some")
        lldbutil.check_variable(
            self,
            optString_Some,
            use_dynamic=False,
            num_children=1)
        uoptString_Some = self.frame().FindVariable("uoptString_Some")
        lldbutil.check_variable(
            self,
            uoptString_Some,
            use_dynamic=False,
            num_children=1)
        uoptString_Some.GetChildAtIndex(99)

        optTrue = self.frame().FindVariable("optTrue")
        lldbutil.check_variable(
            self,
            optTrue,
            use_dynamic=False,
            num_children=1,
            summary='true')

        optFalse = self.frame().FindVariable("optFalse")
        lldbutil.check_variable(
            self,
            optFalse,
            use_dynamic=False,
            num_children=1,
            summary='false')

        optNil = self.frame().FindVariable("optNil")
        lldbutil.check_variable(
            self,
            optNil,
            use_dynamic=False,
            num_children=0,
            summary='nil')

        # Querying a non-existing child should not crash.
        synth_valobj = self.frame().FindVariable("optString_Some")
        synth_valobj.SetSyntheticChildrenGenerated(True);
        self.assertEqual(synth_valobj.GetChildAtIndex(1).GetSummary(), None)
