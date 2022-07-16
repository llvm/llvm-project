import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import re

class TestCase(lldbtest.TestBase):

    @swiftTest
    @skipIf(oslist=['windows', 'linux'])
    def test(self):
        """Test step-in to async functions"""
        self.build()
        src = lldb.SBFileSpec('main.swift')
        _, process, _, _ = lldbutil.run_to_source_breakpoint(self, 'await', src)

	# When run with debug info enabled builds, this prevents stepping from
	# stopping in Swift Concurrency runtime functions.
        self.runCmd("settings set target.process.thread.step-avoid-libraries libswift_Concurrency.dylib")

        # All thread actions are done on the currently selected thread.
        thread = process.GetSelectedThread

        num_async_steps = 0
        while True:
            stop_reason = thread().stop_reason
            if stop_reason == lldb.eStopReasonNone:
                break
            elif stop_reason == lldb.eStopReasonPlanComplete:
                # Run until the next `await` breakpoint.
                process.Continue()
            elif stop_reason == lldb.eStopReasonBreakpoint:
                caller_before = thread().frames[0].function.GetDisplayName()
                line_before = thread().frames[0].line_entry.line
                thread().StepInto()
                caller_after = thread().frames[1].function.GetDisplayName()
                line_after = thread().frames[0].line_entry.line

		# Breakpoints on lines with an `await` may result in more than
		# one breakpoint location. Specifically a location before an
		# async function is called, and then a location on the resume
		# function. In this case, running `step` on these lines will
		# move execution forward within the same function, _not_ step
		# into a new function.
                #
		# As this test is for stepping into async functions, when the
		# step-in keeps execution on the same or next line -- not a
		# different function, then it can be ignored. rdar://76116620
                if line_after in (line_before, line_before + 1):
		    # When stepping stops at breakpoint, don't continue.
                    if thread().stop_reason != lldb.eStopReasonBreakpoint:
                        process.Continue()
                    continue

                # The entry function is missing this prefix dedicating resume functions.
                prefix = re.compile(r'^\([0-9]+\) await resume partial function for ')
                self.assertEqual(prefix.sub('', caller_after),
                                 prefix.sub('', caller_before))
                num_async_steps += 1

        self.assertGreater(num_async_steps, 0)
