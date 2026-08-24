import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftMaterializeUnreadableVariable(lldbtest.TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipEmbeddedSwift
    @swiftTest
    def test_unreadable_variable(self):
        """A frame with two locals: `p`, which reads fine, and `unreadable`, a
        resilient struct with a field whose reflection metadata is missing, so
        its value cannot be read at all.

        All assertions share one launch, since launching the process dominates
        the runtime of this test."""
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self,
            "break here",
            lldb.SBFileSpec("main.swift"),
            extra_images=["Lib", "NoRefl"],
        )
        frame = thread.frames[0]

        # The unreadable variable reports an error, and a readable sibling in
        # the same frame is unaffected.
        lldbutil.check_variable(
            self, frame.FindVariable("p").GetChildMemberWithName("value"), value="7"
        )
        unreadable = frame.FindVariable("unreadable")
        self.assertTrue(unreadable.IsValid(), "'unreadable' was found")
        self.assertTrue(unreadable.GetError().Fail(), "'unreadable' cannot be read")

        # An unreadable variable that the expression does not reference must not
        # make the whole evaluation fail.
        self.expect_expr("p.value", result_value="7")

        # An expression that does need the unreadable variable has to fail and
        # the failure reason given needs to be actionable.
        value = self.frame().EvaluateExpression("unreadable.tag")
        error = value.GetError().GetCString() or ""
        self.assertIn("cannot find 'unreadable' in scope", error)
        self.assertIn('Missing debug information for variable "unreadable"', error)
