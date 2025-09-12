"""
Test that we properly print anonymous members in a base class.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test import decorators


class TestTypeLookupAnonBaseMember(TestBase):
    def test_lookup_anon_base_member(self):
        self.build()
        (target, process, thread, bp1) = lldbutil.run_to_source_breakpoint(
            self, "// Set breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        frame = thread.GetFrameAtIndex(0)

        d = frame.FindVariable("d")
        self.assertTrue(d.IsValid())

        # b from Base
        b = d.GetChildMemberWithName("b")
        self.assertTrue(b.IsValid())
        self.assertEqual(b.GetValueAsSigned(), 1)

        # x from anonymous struct (inside Base)
        x = d.GetChildMemberWithName("x")
        self.assertTrue(x.IsValid())
        self.assertEqual(x.GetValueAsSigned(), 2)

        # d from Derived
        a = d.GetChildMemberWithName("a")
        self.assertTrue(a.IsValid())
        self.assertEqual(a.GetValueAsSigned(), 3)
