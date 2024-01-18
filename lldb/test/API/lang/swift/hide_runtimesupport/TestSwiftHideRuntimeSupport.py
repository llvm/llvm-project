# TestSwiftHideRuntimeSupport.py
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
Test that we hide runtime support values
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftHideRuntimeSupport(TestBase):
    @swiftTest
    def test_swift_hide_runtime_support(self):
        """Test that we hide runtime support values"""

        # This is the function to remove the custom settings in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd(
                'settings set target.display-runtime-support-values true',
                check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("settings set target.display-runtime-support-values false")

        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        self.expect(
            'frame variable -d run',
            substrs=['_0_0'],
            matching=False)
        self.expect('frame variable -d run', substrs=['193627'], matching=True)

        var_opts = lldb.SBVariablesOptions()
        var_opts.SetIncludeArguments(True)
        var_opts.SetIncludeLocals(True)
        var_opts.SetInScopeOnly(True)
        var_opts.SetIncludeStatics(True)
        var_opts.SetIncludeRuntimeSupportValues(False)
        var_opts.SetUseDynamic(lldb.eDynamicCanRunTarget)

        values = self.frame().GetVariables(var_opts)
        found = False
        for value in values:
            if '_0_0' in value.name:
                found = True
            if '$' in value.name:
                found = True
        self.assertFalse(found, "found the thing I was not expecting")

        var_opts.SetIncludeRuntimeSupportValues(True)
        values = self.frame().GetVariables(var_opts)
        found = False
        for value in values:
            if '_0_0' in value.name:
                found = True
        self.assertTrue(found, "not found the thing I was expecting")

        self.runCmd("settings set target.display-runtime-support-values true")
        self.expect(
            'frame variable -d run',
            substrs=['_0_0'],
            matching=True)

        self.runCmd("settings set target.display-runtime-support-values false")
        self.expect(
            'frame variable -d run',
            substrs=['_0_0'],
            matching=False)
