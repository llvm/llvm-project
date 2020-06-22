"""
Tests that importing ObjC modules in a non-ObjC target doesn't crash LLDB.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self,"// break here", lldb.SBFileSpec("main.c"))

        # Import foundation to get some ObjC types.
        self.expect("expr --lang objc -- @import Foundation")
        # Do something with NSString (which requires special handling when
        # preparing to run in the target). The expression most likely can't
        # be prepared to run in the target but it should at least not crash LLDB.
        self.expect('expr --lang objc -- [NSString stringWithFormat:@"%d", 1];',
                    error=True,
                    substrs=["Rewriting an Objective-C constant string requires CFStringCreateWithBytes"])
