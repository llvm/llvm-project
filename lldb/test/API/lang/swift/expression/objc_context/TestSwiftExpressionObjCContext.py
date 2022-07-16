# TestSwiftExpressionObjCContext.py
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
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2

class TestSwiftExpressionObjCContext(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()

        target,  _, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec('main.m'),
            extra_images=['Foo'])

        # This is expected to fail because we can't yet import ObjC
        # modules into a Swift context.
        self.expect("expr -lang Swift -- Bar()", "failure",
                    substrs=["cannot find 'Bar'"],
                    error=True)
        self.expect("expr -lang Swift -- (1, 2, 3)",
                    "context-less swift expression works",
                    substrs=["(Int, Int, Int)"])

