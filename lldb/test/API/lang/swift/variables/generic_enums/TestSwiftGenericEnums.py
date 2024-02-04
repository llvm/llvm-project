# TestSwiftGenericEnums.py
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
Test that we handle reasonably generically-typed enums
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftGenericEnumTypes(TestBase):
    def get_variable(self, name):
        var = self.frame().FindVariable(
            name).GetDynamicValue(lldb.eDynamicCanRunTarget)
        var.SetPreferSyntheticValue(True)
        return var

    @swiftTest
    def test_swift_generic_enum_types(self):
        """Test that we handle reasonably generically-typed enums"""
        self.build()
        main_file = lldb.SBFileSpec('main.swift')
        target, process, thread, bkpt1 = lldbutil.run_to_source_breakpoint(
            self, 'Set first breakpoint here', main_file)
        bkpt2 = target.BreakpointCreateBySourceRegex(
            'Set second breakpoint here', main_file)
        self.assertGreater(bkpt2.GetNumLocations(), 0, VALID_BREAKPOINT)


        enumvar = self.get_variable("myOptionalU").GetStaticValue()
        self.assertTrue(enumvar.GetValue() is None,
                        "static type has a value when it shouldn't")
        enumvar = enumvar.GetDynamicValue(lldb.eDynamicCanRunTarget)
        # FIXME?
        #self.assertEqual(
        #    enumvar.GetValue(), "Some",
        #    "dynamic type's value should be Some")
        self.assertEqual(
            enumvar.GetSummary(), "3",
            "Some's summary should be 3")

        threads = lldbutil.continue_to_breakpoint(process, bkpt2)
        self.assertEqual(len(threads), 1)

        value = self.get_variable("value")
        lldbutil.check_variable(
            self,
            value,
            use_dynamic=True,
            summary='"Now with Content"',
            typename='Swift.Optional<Swift.String>')

