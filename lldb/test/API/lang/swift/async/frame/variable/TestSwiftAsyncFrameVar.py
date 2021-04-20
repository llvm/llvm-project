import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil

class TestCase(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    @skipIf(oslist=['windows', 'linux'])
    def test(self):
        """Test `frame variable` in async functions"""
        self.build()

        # Setting a breakpoint on "inner" results in a breakpoint at the start
        # of each coroutine "funclet" function.
        _, process, _, _ = lldbutil.run_to_name_breakpoint(self, 'inner')

        # The step-over actions in the below commands may not be needed in the
        # future, but for now they are. This comment describes why. Take the
        # following line of code:
        #     let x = await asyncFunc()
        # Some breakpoints, including the ones in this test, resolve to
        # locations that are at the start of resume functions. At the start of
        # the resume function, the assignment may not be complete. In order to
        # ensure assignment takes place, step-over is used to take execution to
        # the next line.

        stop_num = 0
        while process.state == lldb.eStateStopped:
            thread = process.GetSelectedThread()
            frame = thread.frames[0]
            if stop_num == 0:
                # Start of the function.
                pass
            elif stop_num == 1:
                # After first await, read `a`.
                a = frame.FindVariable("a")
                self.assertTrue(a.IsValid())
                self.assertEqual(a.unsigned, 0)
                # Step to complete `a`'s assignment (stored in the stack).
                thread.StepOver()
                self.assertGreater(a.unsigned, 0)
            elif stop_num == 2:
                # After second, read `a` and `b`.
                # At this point, `a` can be read from the async context.
                a = frame.FindVariable("a")
                self.assertTrue(a.IsValid())
                self.assertGreater(a.unsigned, 0)
                b = frame.FindVariable("b")
                self.assertTrue(b.IsValid())
                self.assertEqual(b.unsigned, 0)
                # Step to complete `b`'s assignment (stored in the stack).
                thread.StepOver()
                self.assertGreater(b.unsigned, 0)
            else:
                # Unexpected stop.
                self.assertTrue(False)

            stop_num += 1
            process.Continue()
