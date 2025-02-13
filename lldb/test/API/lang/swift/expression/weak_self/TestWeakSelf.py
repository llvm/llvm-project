# TestWeakSelf.py
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
#
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *	
import lldbsuite.test.lldbutil as lldbutil



class TestSwiftGenericExtension(TestBase):
     @swiftTest
     def test(self):
        self.build()

        target, process, _, _ = lldbutil.run_to_source_breakpoint(
             self, "break here for if let success", lldb.SBFileSpec("main.swift")
         )
        self.expect("expr self", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["ClosureMaker)", "5"])
        self.expect("expr self", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["ClosureMaker)", "5"])
        self.expect("expr a", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["5"])

        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here for guard let success', lldb.SBFileSpec('main.swift'), None)
        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("expr self", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["ClosureMaker)", "5"])
        self.expect("expr a", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["5"])


        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here for if let else', lldb.SBFileSpec('main.swift'), None)
        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("expr self", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["nil"])

        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here for guard let else', lldb.SBFileSpec('main.swift'), None)
        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("expr self", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["nil"])
