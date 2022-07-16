# TestSwiftMetadataSymbols.py
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
Test that swift metadata symbols are printed and classified correctly
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os


class TestSwiftMetadataSymbols(TestBase):

    @swiftTest
    @expectedFailureAll(bugnumber="<rdar://problem/31066543>")
    def test_swift_metadata_symbols(self):
        """Test swift Class types"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)

    def do_test(self):
        """Tests that we can break and display simple types"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        self.expect("target modules dump symtab -m a.out",
                    patterns=['Metadata.*_TMC1a3Foo'])

        self.expect("target modules dump symtab a.out",
                    patterns=['Metadata.*type metadata for'])


