# TestObjCImportedTypes.py
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
"""n
Test that we are able to deal with ObjC-imported types
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftObjCImportedTypes(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    @expectedFailureAll(bugnumber="rdar://60396797",
                        setting=('symbols.use-swift-clangimporter', 'false'))
    @swiftTest
    @skipUnlessDarwin
    def test_swift_objc_imported_types(self):
        """Test that we are able to deal with ObjC-imported types"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        nss = self.frame().FindVariable("nss")
        nsn = self.frame().FindVariable("nsn")
        nsmo = self.frame().FindVariable("nsmo")
        nsmd = self.frame().FindVariable("nsmd")

        lldbutil.check_variable(
            self,
            nss,
            use_dynamic=False,
            typename="Foundation.NSString")
        lldbutil.check_variable(
            self,
            nsn,
            use_dynamic=False,
            typename="Foundation.NSNumber")
        lldbutil.check_variable(
            self,
            nsmo,
            use_dynamic=False,
            typename="CoreData.NSManagedObject")
        lldbutil.check_variable(
            self,
            nsmd,
            use_dynamic=False,
            typename="Foundation.NSMutableDictionary")

        lldbutil.check_variable(self, nss, use_dynamic=True, summary='"abc"')
        lldbutil.check_variable(self, nsn, use_dynamic=True, summary='Int64(3)')
        lldbutil.check_variable(
            self,
            nsmo,
            use_dynamic=True,
            typename='CoreData.NSManagedObject')
        lldbutil.check_variable(
            self,
            nsmd,
            use_dynamic=True,
            summary='1 key/value pair')

