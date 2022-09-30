"""Test that SBFrame::GetExpressionPath construct valid expressions"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SBValueGetExpressionPathTest(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def path(self, value):
        """Constructs the expression path given the SBValue"""
        if not value:
            return None
        stream = lldb.SBStream()
        if not value.GetExpressionPath(stream):
            return None
        return stream.GetData()

    def test_expression_path(self):
        """Test that SBFrame::GetExpressionPath construct valid expressions"""
        self.build()
        self.setTearDownCleanup()

        exe = self.getBuildArtifact("a.out")

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here', lldb.SBFileSpec("main.cpp"))
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertEquals(len(threads), 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        # Find "b" variables in frame
        b = self.frame.FindVariable("b")
        bp = self.frame.FindVariable("b_ptr")
        br = self.frame.FindVariable("b_ref")
        bpr = self.frame.FindVariable("b_ptr_ref")
        # Check expression paths
        self.assertEqual(self.path(b), "b")
        self.assertEqual(self.path(bp), "b_ptr")
        self.assertEqual(self.path(br), "b_ref")
        self.assertEqual(self.path(bpr), "b_ptr_ref")

        # Dereference "b" pointers
        bp_deref = bp.Dereference()
        bpr_deref = bpr.Dereference()  # a pointer
        bpr_deref2 = bpr_deref.Dereference()  # two Dereference() calls to get object
        # Check expression paths
        self.assertEqual(self.path(bp_deref), "*(b_ptr)")
        self.assertEqual(self.path(bpr_deref), "b_ptr_ref")
        self.assertEqual(self.path(bpr_deref2), "*(b_ptr_ref)")

        # Access "b" members and check expression paths
        self.assertEqual(self.path(b.GetChildMemberWithName("x")), "b.x")
        self.assertEqual(self.path(bp.GetChildMemberWithName("x")), "b_ptr->x")
        self.assertEqual(self.path(br.GetChildMemberWithName("x")), "b_ref.x")
        self.assertEqual(self.path(bp_deref.GetChildMemberWithName("x")), "(*(b_ptr)).x")
        self.assertEqual(self.path(bpr_deref.GetChildMemberWithName("x")), "b_ptr_ref->x")
        self.assertEqual(self.path(bpr_deref2.GetChildMemberWithName("x")), "(*(b_ptr_ref)).x")
        # TODO: Uncomment once accessing members on pointer references is supported.
        # self.assertEqual(self.path(bpr.GetChildMemberWithName("x")), "b_ptr_ref->x")

        # Try few expressions with multiple member access
        bp_ar_x = bp.GetChildMemberWithName("a_ref").GetChildMemberWithName("x")
        br_ar_y = br.GetChildMemberWithName("a_ref").GetChildMemberWithName("y")
        self.assertEqual(self.path(bp_ar_x), "b_ptr->a_ref.x")
        self.assertEqual(self.path(br_ar_y), "b_ref.a_ref.y")
        bpr_deref_apr_deref = bpr_deref.GetChildMemberWithName("a_ptr_ref").Dereference()
        bpr_deref_apr_deref2 = bpr_deref_apr_deref.Dereference()
        self.assertEqual(self.path(bpr_deref_apr_deref), "b_ptr_ref->a_ptr_ref")
        self.assertEqual(self.path(bpr_deref_apr_deref2), "*(b_ptr_ref->a_ptr_ref)")
        bpr_deref_apr_deref_x = bpr_deref_apr_deref.GetChildMemberWithName("x")
        bpr_deref_apr_deref2_x = bpr_deref_apr_deref2.GetChildMemberWithName("x")
        self.assertEqual(self.path(bpr_deref_apr_deref_x), "b_ptr_ref->a_ptr_ref->x")
        self.assertEqual(self.path(bpr_deref_apr_deref2_x), "(*(b_ptr_ref->a_ptr_ref)).x")

        # Find "c" variables in frame
        c = self.frame.FindVariable("c")
        cp = self.frame.FindVariable("c_ptr")
        cr = self.frame.FindVariable("c_ref")
        cpr = self.frame.FindVariable("c_ptr_ref")
        # Dereference pointers
        cp_deref = cp.Dereference()
        cpr_deref = cpr.Dereference()  # a pointer
        cpr_deref2 = cpr_deref.Dereference()  # two Dereference() calls to get object
        # Check expression paths
        self.assertEqual(self.path(cp_deref), "*(c_ptr)")
        self.assertEqual(self.path(cpr_deref), "c_ptr_ref")
        self.assertEqual(self.path(cpr_deref2), "*(c_ptr_ref)")

        # Access members on "c" variables and check expression paths
        self.assertEqual(self.path(c.GetChildMemberWithName("x")), "c.x")
        self.assertEqual(self.path(cp.GetChildMemberWithName("x")), "c_ptr->x")
        self.assertEqual(self.path(cr.GetChildMemberWithName("x")), "c_ref.x")
        self.assertEqual(self.path(cp_deref.GetChildMemberWithName("x")), "(*(c_ptr)).x")
        self.assertEqual(self.path(cpr_deref.GetChildMemberWithName("x")), "c_ptr_ref->x")
        self.assertEqual(self.path(cpr_deref2.GetChildMemberWithName("x")), "(*(c_ptr_ref)).x")
        # TODO: Uncomment once accessing members on pointer references is supported.
        # self.assertEqual(self.path(cpr.GetChildMemberWithName("x")), "c_ptr_ref->x")