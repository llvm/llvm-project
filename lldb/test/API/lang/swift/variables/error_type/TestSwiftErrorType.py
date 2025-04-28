# TestSwiftErrorType.py
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
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftErrorType(TestBase):
    @swiftTest
    def test(self):
        """Test handling of Swift Error types"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        self.expect('frame variable -d run -- s', substrs=['SError','payload = 1'])
        self.expect('frame variable -d run -- c', substrs=['CError','payload = 2'])
        self.expect('frame variable -d run -- e', substrs=['EError','OutOfCookies'])
        self.expect('expr -d run -- s', substrs=['SError','payload = 1'])
        self.expect('expr -d run -- c', substrs=['CError','payload = 2'])
        self.expect('expr -d run -- e', substrs=['EError','OutOfCookies'])

