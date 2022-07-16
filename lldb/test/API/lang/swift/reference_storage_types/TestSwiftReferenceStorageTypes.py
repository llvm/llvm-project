# TestSwiftReferenceStorageTypes.py
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
Test weak, unowned and unmanaged types
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftReferenceStorageTypes(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    @decorators.skipIf(archs=['ppc64le']) #SR-10215
    @swiftTest
    @skipIf(oslist=["linux"], bugnumber="rdar://76592966")
    def test_swift_reference_storage_types(self):
        """Test weak, unowned and unmanaged types"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect('frame variable myclass.sub_001', substrs=['x = 1'])
        self.expect('frame variable myclass.sub_002', substrs=['x = 1'])
        self.expect('frame variable myclass.sub_003', substrs=['x = 1'])
        self.expect('frame variable myclass.sub_004', substrs=['x = 1'])
        self.expect('frame variable myclass.sub_005', substrs=['x = 1'])
        self.expect('frame variable myclass.sub_006', substrs=['x = 1'])

        self.expect('expression myclass.sub_001', substrs=['x = 1'])
        self.expect('expression myclass.sub_002', substrs=['x = 1'])
        self.expect('expression myclass.sub_003', substrs=['x = 1'])
        self.expect('expression myclass.sub_004', substrs=['x = 1'])
        self.expect('expression myclass.sub_005', substrs=['x = 1'])
        self.expect('expression myclass.sub_006', substrs=['x = 1'])

        self.expect('expression myclass.sub_001!', substrs=['x = 1'])
        self.expect('expression myclass.sub_002!', substrs=['x = 1'])
        self.expect('expression myclass.sub_003!', substrs=['x = 1'])
        self.expect('expression myclass.sub_004!', substrs=['x = 1'])

        myclass = self.frame().FindVariable("myclass")
        sub_001 = myclass.GetChildMemberWithName("sub_001")
        sub_002 = myclass.GetChildMemberWithName("sub_002")
        sub_003 = myclass.GetChildMemberWithName("sub_003")
        sub_004 = myclass.GetChildMemberWithName("sub_004")
        sub_005 = myclass.GetChildMemberWithName("sub_005")
        sub_006 = myclass.GetChildMemberWithName("sub_006")

        sub_001_type = sub_001.GetType()
        sub_002_type = sub_002.GetType()
        sub_003_type = sub_003.GetType()
        sub_004_type = sub_004.GetType()
        sub_005_type = sub_005.GetType()
        sub_006_type = sub_006.GetType()

        self.assertTrue(sub_001_type.IsValid(), "001.GetType() is valid")
        self.assertTrue(sub_002_type.IsValid(), "002.GetType() is valid")
        self.assertTrue(sub_003_type.IsValid(), "003.GetType() is valid")
        self.assertTrue(sub_004_type.IsValid(), "004.GetType() is valid")
        self.assertTrue(sub_005_type.IsValid(), "005.GetType() is valid")
        self.assertTrue(sub_006_type.IsValid(), "006.GetType() is valid")

        sub_001_type.GetTypeClass()
        sub_002_type.GetTypeClass()
        sub_003_type.GetTypeClass()
        sub_004_type.GetTypeClass()
        sub_005_type.GetTypeClass()
        sub_006_type.GetTypeClass()
