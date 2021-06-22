"""
Test SB API support for identifying artificial (tail call) frames.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

class TestTailCallFrameSBAPI(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    @skipIf(compiler="clang", compiler_version=['<', '10.0'])
    @skipIf(dwarf_version=['<', '4'])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr26265")
    def test_tail_call_frame_sbapi(self):
        self.build()
        self.do_test()

    def do_test(self):
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateBySourceRegex("break here",
                lldb.SBFileSpec("main.cpp"))
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        error = lldb.SBError()
        launch_info = target.GetLaunchInfo()
        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        # Did we hit our breakpoint?
        threads = lldbutil.get_threads_stopped_at_breakpoint(process,
                breakpoint)
        self.assertEqual(
            len(threads), 1,
            "There should be a thread stopped at our breakpoint")

        self.assertEqual(breakpoint.GetHitCount(), 1)

        thread = threads[0]

        # Here's what we expect to see in the backtrace:
        #   frame #0: ... a.out`sink() at main.cpp:13:4 [opt]
        #   frame #1: ... a.out`func3() at main.cpp:14:1 [opt] [artificial]
        #   frame #2: ... a.out`func2() at main.cpp:18:62 [opt]
        #   frame #3: ... a.out`func1() at main.cpp:18:85 [opt] [artificial]
        #   frame #4: ... a.out`main at main.cpp:23:3 [opt]
        names = ["sink", "func3", "func2", "func1", "main"]
        artificiality = [False, True, False, True, False]
        for idx, (name, is_artificial) in enumerate(zip(names, artificiality)):
            frame = thread.GetFrameAtIndex(idx)

            # Use a relaxed substring check because function dislpay names are
            # platform-dependent. E.g we see "void sink(void)" on Windows, but
            # "sink()" on Darwin. This seems like a bug -- just work around it
            # for now.
            self.assertIn(name, frame.GetDisplayFunctionName())
            self.assertEqual(frame.IsArtificial(), is_artificial)
