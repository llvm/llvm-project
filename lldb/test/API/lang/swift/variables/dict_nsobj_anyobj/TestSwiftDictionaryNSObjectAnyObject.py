# TestSwiftDictionaryNSObjectAnyObject.py
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
Tests that we properly vend synthetic children for Swift.Dictionary<NSObject,AnyObject>
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestDictionaryNSObjectAnyObject(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test_dictionary_nsobject_any_object(self):
        """Tests that we properly vend synthetic children for Swift.Dictionary<NSObject,AnyObject>"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        if self.getArchitecture() in ['arm', 'armv7', 'armv7k', 'i386']:
            self.expect(
                "frame variable -d run -- d2",
                ordered=False,
                substrs=[
                    'Int32(1)',
                    'Int32(2)',
                    'Int32(3)',
                    'Int32(4)'])
        else:
            self.expect(
                "frame variable -d run -- d2",
                ordered=False,
                substrs=[
                    'Int64(1)',
                    'Int64(2)',
                    'Int64(3)',
                    'Int64(4)'])

        self.expect(
            "frame variable -d run -- d3",
            substrs=[
                'key = "hello"',
                'value = 123'])

