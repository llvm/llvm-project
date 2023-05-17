"""
Make sure running internal expressions doesn't
influence the result variable numbering.
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestExpressionResultNumbering(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    def test_sample_rename_this(self):
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.do_numbering_test()

    def do_numbering_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint here", self.main_source_file)

        bkpt = target.BreakpointCreateBySourceRegex("Add conditions to this breakpoint",
                                                    self.main_source_file)
        self.assertEqual(bkpt.GetNumLocations(), 1, "Set the breakpoint")

        bkpt.SetCondition("call_me(value) < 6")

        # Get the number of the last expression:
        result = thread.frames[0].EvaluateExpression("call_me(200)")
        self.assertSuccess(result.GetError(), "Our expression succeeded")
        name = result.GetName()
        ordinal = int(name[1:])
        
        process.Continue()

        # The condition evaluation had to run a 4 expressions, but we haven't
        # run any user expressions.
        result = thread.frames[0].EvaluateExpression("call_me(200)")
        self.assertSuccess(result.GetError(), "Our expression succeeded the second time")
        after_name = result.GetName()
        after_ordinal = int(after_name[1:])
        self.assertEqual(ordinal + 1, after_ordinal) 
