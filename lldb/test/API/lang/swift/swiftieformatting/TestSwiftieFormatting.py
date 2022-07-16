# TestSwiftieFormatting.py
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
Test that data formatters honor Swift conventions
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftieFormatting(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    @skipUnlessDarwin
    @swiftTest
    def test_swiftie_formatting(self):
        """Test that data formatters honor Swift conventions"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        swcla = self.frame().FindVariable("swcla")
        swcla.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        swcla.SetPreferSyntheticValue(True)

        ns_a = swcla.GetChildMemberWithName("ns_a")
        self.assertTrue(
            ns_a.GetSummary() == '"Hello Swift"',
            "ns_a summary wrong")

        ns_d = swcla.GetChildMemberWithName("ns_d")
        self.assertTrue(ns_d.GetSummary() == '0 bytes', "ns_d summary wrong")

        IntWidth = 64
        if self.getArchitecture() in ['arm', 'armv7', 'armv7k', 'i386']:
            IntWidth = 32

        ns_n = swcla.GetChildMemberWithName("ns_n")
        self.assertTrue(ns_n.GetSummary() == ("Int%d(30)" % IntWidth), "ns_n summary wrong")

        ns_u = swcla.GetChildMemberWithName("ns_u")
        self.assertTrue(ns_u.GetSummary() == ('"page.html -- http://www.apple.com"'), "ns_u summary wrong")

        swcla = self.frame().EvaluateExpression("swcla")
        swcla.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        swcla.SetPreferSyntheticValue(True)

        ns_a = swcla.GetChildMemberWithName("ns_a")
        self.assertTrue(
            ns_a.GetSummary() == '"Hello Swift"',
            "ns_a summary wrong")

        ns_d = swcla.GetChildMemberWithName("ns_d")
        self.assertTrue(ns_d.GetSummary() == '0 bytes', "ns_d summary wrong")

        ns_n = swcla.GetChildMemberWithName("ns_n")
        self.assertTrue(ns_n.GetSummary() == ("Int%d(30)" % IntWidth), "ns_n summary wrong")

        ns_u = swcla.GetChildMemberWithName("ns_u")
        self.assertTrue(ns_u.GetSummary() == ('"page.html -- http://www.apple.com"'), "ns_u summary wrong")

        nsarr = self.frame().FindVariable("nsarr")
        nsarr.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        nsarr.SetPreferSyntheticValue(True)

        nsarr0 = nsarr.GetChildAtIndex(0)
        nsarr0.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        nsarr0.SetPreferSyntheticValue(True)
        nsarr1 = nsarr.GetChildAtIndex(1)
        nsarr1.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        nsarr1.SetPreferSyntheticValue(True)
        nsarr3 = nsarr.GetChildAtIndex(3)
        nsarr3.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        nsarr3.SetPreferSyntheticValue(True)

        self.assertTrue(
            nsarr0.GetSummary() == ("Int%d(2)" % IntWidth),
            'nsarr[0] summary wrong')
        self.assertTrue(
            nsarr1.GetSummary() == ("Int%d(3)" % IntWidth),
            'nsarr[1] summary wrong')
        self.assertTrue(
            nsarr3.GetSummary() == ("Int%d(5)" % IntWidth),
            'nsarr[3] summary wrong')

        self.expect(
            'frame variable -d run nsarr[4] --ptr-depth=1',
            substrs=[
                '"One"',
                '"Two"',
                '"Three"'])
        self.expect(
            'frame variable -d run nsarr[5] --ptr-depth=1',
            substrs=[
                ("Int%d(1)" % IntWidth),
                ("Int%d(2)" % IntWidth),
                ("Int%d(3)" % IntWidth)])
