# TestEqualityOperators.py
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
Test that we resolve various shadowed equality operators properly
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import no_match
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import unittest2

import sys
if sys.version_info.major == 2:
    import commands as subprocess
else:
    import subprocess

def execute_command(command):
    (exit_status, output) = subprocess.getstatusoutput(command)
    return exit_status


class TestUnitTests(TestBase):

    @swiftTest
    @skipIf(debug_info=no_match(["dsym"]), bugnumber="This test is building a dSYM")
    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="rdar://28180489")
    def test_equality_operators_fileprivate(self):
        """Test that we resolve expression operators correctly"""
        self.build()
        self.do_test("Fooey.CompareEm1", "true", 1)

    @swiftTest
    @skipIf(debug_info=no_match(["dsym"]), bugnumber="This test is building a dSYM")
    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="rdar://28180489")
    def test_equality_operators_private(self):
        """Test that we resolve expression operators correctly"""
        self.build()
        self.do_test("Fooey.CompareEm2", "false", 2)

    @swiftTest
    @skipIf(debug_info=no_match(["dsym"]), bugnumber="This test is building a dSYM")
    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="rdar://28180489")
    @expectedFailureAll(bugnumber="rdar://38483465")
    def test_equality_operators_other_module(self):
        """Test that we resolve expression operators correctly"""
        self.build()
        self.do_test("Fooey.CompareEm3", "false", 3)

    def setUp(self):
        TestBase.setUp(self)

    def do_test(self, bkpt_name, compare_value, counter_value):
        """Test that we resolve expression operators correctly"""
        lldbutil.run_to_name_breakpoint(self, bkpt_name,
                                        exe_name=self.getBuildArtifact("three"),
                                        extra_images=["fooey"])

        options = lldb.SBExpressionOptions()

        value = self.frame().EvaluateExpression("lhs == rhs", options)
        self.assertSuccess(
            value.GetError(),
            "Expression in %s was successful" % bkpt_name)
        summary = value.GetSummary()
        self.assertTrue(
            summary == compare_value,
            "Expression in CompareEm has wrong value: %s (expected %s)." %
            (summary,
             compare_value))

        # And make sure we got did increment the counter by the right value.
        value = self.frame().EvaluateExpression("Fooey.GetCounter()", options)
        self.assertSuccess(value.GetError(), "GetCounter expression failed")

        counter = value.GetValueAsUnsigned()
        self.assertTrue(
            counter == counter_value, "Counter value is wrong: %d (expected %d)" %
            (counter, counter_value))

        # Make sure the presence of these type specific == operators doesn't interfere
        # with finding other unrelated == operators.
        value = self.frame().EvaluateExpression("1 == 2", options)
        self.assertSuccess(value.GetError(), "1 == 2 expression couldn't run")
        self.assertTrue(
            value.GetSummary() == "false",
            "1 == 2 didn't return false.")

