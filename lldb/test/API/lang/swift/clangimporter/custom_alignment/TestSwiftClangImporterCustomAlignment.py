# TestSwiftClangImporterCustomAlignment.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2019 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os


class TestSwiftClangImporterCustomAlignment(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    def test(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.frames[0]
        v = frame.FindVariable("v")
        s = v.GetChildMemberWithName("s")

        field_1 = s.GetChildMemberWithName("field_64_1")
        lldbutil.check_variable(self, field_1, False, value="100")

        field_2 = s.GetChildMemberWithName("field_32_1")
        lldbutil.check_variable(self, field_2, False, value="200")

        field_3 = s.GetChildMemberWithName("field_32_2")
        lldbutil.check_variable(self, field_3, False, value="300")

        field_4 = s.GetChildMemberWithName("field_64_2")
        lldbutil.check_variable(self, field_4, False, value="400")

        x = v.GetChildMemberWithName("x")
        lldbutil.check_variable(self, x, False, value="1")
