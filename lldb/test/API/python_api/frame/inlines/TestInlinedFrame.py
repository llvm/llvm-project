"""
Testlldb Python SBFrame APIs IsInlined() and GetFunctionName().
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class InlinedFrameAPITestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.source = "inlines.c"
        self.first_stop = line_number(
            self.source, "// This should correspond to the first break stop."
        )
        self.second_stop = line_number(
            self.source, "// This should correspond to the second break stop."
        )

    def test_stop_at_outer_inline(self):
        """Exercise SBFrame.IsInlined() and SBFrame.GetFunctionName()."""
        self.build()
        _, process, thread, _ = lldbutil.run_to_name_breakpoint(
            self, "inner_inline", bkpt_module="a.out"
        )

        stack_traces1 = lldbutil.print_stacktraces(process, string_buffer=True)
        if self.TraceOn():
            print(
                "Full stack traces when first stopped on the breakpoint 'inner_inline':"
            )
            print(stack_traces1)

        # The first breakpoint should correspond to an inlined call frame.
        # If it's an inlined call frame, expect to find, in the stack trace,
        # that there is a frame which corresponds to the following call site:
        #
        #     outer_inline (argc);
        #
        frame0 = thread.GetFrameAtIndex(0)
        if frame0.IsInlined():
            filename = frame0.GetLineEntry().GetFileSpec().GetFilename()
            self.assertEqual(filename, self.source)
            self.expect(
                stack_traces1,
                "First stop at %s:%d" % (self.source, self.first_stop),
                exe=False,
                substrs=["%s:%d" % (self.source, self.first_stop)],
            )

            # Expect to break again for the second time.
            process.Continue()
            self.assertState(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
            stack_traces2 = lldbutil.print_stacktraces(process, string_buffer=True)
            if self.TraceOn():
                print(
                    "Full stack traces when stopped on the breakpoint 'inner_inline' for the second time:"
                )
                print(stack_traces2)
                self.expect(
                    stack_traces2,
                    "Second stop at %s:%d" % (self.source, self.second_stop),
                    exe=False,
                    substrs=["%s:%d" % (self.source, self.second_stop)],
                )
