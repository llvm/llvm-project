# TestSwiftPOValTypes.py
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
import lldbsuite.test
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class TestSwiftPOValueTypes(TestBase):

    @swiftTest
    def test_value_types(self):
        """Test 'po' on a variety of value types with and without custom descriptions."""
        self.build()
        (_,_,_,_) = lldbutil.run_to_source_breakpoint(self, "Break here to run tests", lldb.SBFileSpec("main.swift"))
        
        self.expect("po dm", substrs=['a', '12', 'b', '24'])
        self.expect("po cm", substrs=['c', '36'])
        self.expect("po cm", substrs=['12', '24'], matching=False)
        self.expect("po cs", substrs=['CustomDebugStringConvertible'])
        self.expect("po cs", substrs=['CustomStringConvertible'], matching=False)
        self.expect("po cs", substrs=['a', '12', 'b', '24'])
        self.expect("script lldb.frame.FindVariable('cs').GetObjectDescription()", substrs=['a', '12', 'b', '24'])
        self.expect("po (12,24,36,48)", substrs=['12', '24', '36', '48'])
        self.expect("po (dm as Any, cm as Any,48 as Any)", substrs=['12', '24', '36', '48'])
        self.expect("po patatino", substrs=['foo'])

    @swiftTest
    def test_ignore_bkpts_in_po(self):
        """Run a po expression with a breakpoint in the debugDescription, make sure we don't hit it."""

        self.build()
        main_spec = lldb.SBFileSpec("main.swift")
        (target, process, thread, _) = lldbutil.run_to_source_breakpoint(self, "Break here to run tests", main_spec)
        po_bkpt = target.BreakpointCreateBySourceRegex("Breakpoint in debugDescription", main_spec)

        # As part of the po expression we should auto-continue past the breakpoint so this succeeds:
        self.expect("po cs", substrs=['CustomDebugStringConvertible'])
        self.assertEqual(po_bkpt.GetHitCount(), 1, "Did hit the breakpoint")

        
        
