# TestSwiftBacktracePrinting.py
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
Test printing Swift backtrace
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import os
import unittest2


class TestSwiftBacktracePrinting(TestBase):
    @swiftTest
    def test_swift_backtrace_printing(self):
        """Test printing Swift backtrace"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        self.expect("bt", substrs=['h<Int>',
                                   'g<String, Int>', 'pair', # FIXME: values are still wrong!
                                   'arg1=12', 'arg2="Hello world"'])
        self.expect("breakpoint set -p other", substrs=['g<U, T>'])

