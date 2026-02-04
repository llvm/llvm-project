# TestClosureShortcuts.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2018 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestClosureShortcuts(TestBase):
    @swiftTest
    def test(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here for anonymous variable", lldb.SBFileSpec("main.swift")
        )
        #
        # rdar://159316245
        self.runCmd("settings set target.experimental.use-DIL false")
        self.expect("expr $0", substrs=["patatino"])
        self.expect("expr $1", substrs=["foo"])
        self.expect("frame var $0", substrs=["patatino"])
        self.expect("frame var $1", substrs=["foo"])

        lldbutil.continue_to_source_breakpoint(
            self, process, "break here for tinky", lldb.SBFileSpec("main.swift")
        )
        self.expect("expr [12, 14].map({$0 + 2})", substrs=["[0] = 14", "[1] = 16"])

        lldbutil.continue_to_source_breakpoint(
            self, process, "break here for outer scope", lldb.SBFileSpec("main.swift")
        )
        self.expect("expr tinky.map({$0 * 2})", substrs=["[0] = 4", "[1] = 8"])
        self.expect("expr [2,4].map({$0 * 2})", substrs=["[0] = 4", "[1] = 8"])
        self.expect("expr $0", substrs=["anonymous closure argument not contained in a closure"], error=True)
