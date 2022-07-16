# TestSwiftBackwardsCompatibilitySimple.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2018 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test import configuration
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import swift
import unittest2


class TestSwiftBackwardsCompatibilitySimple(lldbtest.TestBase):

    @swiftTest
    @skipIf(compiler="swiftc", compiler_version=['<', '5.0'])
    def test_simple(self):
        if configuration.swiftCompiler:
            compiler = configuration.swiftCompiler
        else:
            compiler = swift.getSwiftCompiler()
        version = self.getCompilerVersion(compiler)
        if version < '5.0':
            self.skipTest('Swift compiler predates stable ABI')
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.swift'))
        # FIXME: Removing the next line breaks subsequent expressions
        #        when the swiftmodules can't be loaded.
        self.expect("fr v")
        self.expect("fr v number", substrs=['23'])
        self.expect("fr v array", substrs=['1', '2', '3'])
        self.expect("fr v string", substrs=['"hello"'])
        self.expect("fr v tuple", substrs=['42', '"abc"'])
        # FIXME: This doesn't work.
        #self.expect("fr v generic", substrs=['-1'])

