"""
Test that our use of "Optional" to wrap expressions in a weak self
context doesn't collide with names inherited from ObjC.
"""

from __future__ import print_function

import os
import time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestOptionalAmbiguity(TestBase):
    @swiftTest
    def test_sample_rename_this(self):
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.swift")
        self.print_in_closure()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def print_in_closure(self):
        (_, _, thread, _) = lldbutil.run_to_source_breakpoint(self,
                                "Set a breakpoint here", self.main_source_file) 

        # Since the failure mode of a name collision is that the expression fails
        # to compile, I just check that here.
        frame = thread.GetFrameAtIndex(0)
        ret_value = frame.EvaluateExpression("self")
        self.assertSuccess(ret_value.GetError(), "The expression completed successfully")

