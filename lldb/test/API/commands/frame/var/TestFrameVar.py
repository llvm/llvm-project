"""
Make sure the frame variable -g, -a, and -l flags work.
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestFrameVar(TestBase):

    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_var(self):
        self.build()
        self.do_test()

    def do_test(self):
        target = self.createTestTarget()

        # Now create a breakpoint in main.c at the source matching
        # "Set a breakpoint here"
        breakpoint = target.BreakpointCreateBySourceRegex(
            "Set a breakpoint here", lldb.SBFileSpec("main.c"))
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() >= 1,
                        VALID_BREAKPOINT)

        error = lldb.SBError()
        # This is the launch info.  If you want to launch with arguments or
        # environment variables, add them using SetArguments or
        # SetEnvironmentEntries

        launch_info = target.GetLaunchInfo()
        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        # Did we hit our breakpoint?
        from lldbsuite.test.lldbutil import get_threads_stopped_at_breakpoint
        threads = get_threads_stopped_at_breakpoint(process, breakpoint)
        self.assertEqual(
            len(threads), 1,
            "There should be a thread stopped at our breakpoint")

        # The hit count for the breakpoint should be 1.
        self.assertEquals(breakpoint.GetHitCount(), 1)

        frame = threads[0].GetFrameAtIndex(0)
        command_result = lldb.SBCommandReturnObject()
        interp = self.dbg.GetCommandInterpreter()

        # Just get args:
        result = interp.HandleCommand("frame var -l", command_result)
        self.assertEqual(result, lldb.eReturnStatusSuccessFinishResult, "frame var -a didn't succeed")
        output = command_result.GetOutput()
        self.assertIn("argc", output, "Args didn't find argc")
        self.assertIn("argv", output, "Args didn't find argv")
        self.assertNotIn("test_var", output, "Args found a local")
        self.assertNotIn("g_var", output, "Args found a global")

        # Just get locals:
        result = interp.HandleCommand("frame var -a", command_result)
        self.assertEqual(result, lldb.eReturnStatusSuccessFinishResult, "frame var -a didn't succeed")
        output = command_result.GetOutput()
        self.assertNotIn("argc", output, "Locals found argc")
        self.assertNotIn("argv", output, "Locals found argv")
        self.assertIn("test_var", output, "Locals didn't find test_var")
        self.assertNotIn("g_var", output, "Locals found a global")

        # Get the file statics:
        result = interp.HandleCommand("frame var -l -a -g", command_result)
        self.assertEqual(result, lldb.eReturnStatusSuccessFinishResult, "frame var -a didn't succeed")
        output = command_result.GetOutput()
        self.assertNotIn("argc", output, "Globals found argc")
        self.assertNotIn("argv", output, "Globals found argv")
        self.assertNotIn("test_var", output, "Globals found test_var")
        self.assertIn("g_var", output, "Globals didn't find g_var")



