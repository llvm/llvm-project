"""Test that backtraces can follow cross-DSO tail calls"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCrossDSOTailCalls(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    @skipIf(compiler="clang", compiler_version=['<', '8.0'])
    @skipIf(dwarf_version=['<', '4'])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr26265")
    @expectedFailureAll(archs=['arm', 'aarch64'], bugnumber="llvm.org/PR44561")
    def test_cross_dso_tail_calls(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Register our shared libraries for remote targets so they get
        # automatically uploaded
        environment = self.registerSharedLibrariesWithTarget(
            target, ['One', 'Two'])

        lldbutil.run_break_set_by_source_regexp(self, '// break here',
                extra_options='-f Two.c')

        process = target.LaunchSimple(
            None, environment, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # We should be stopped in the second dylib.
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)

        # Debug helper:
        # self.runCmd("log enable -f /tmp/lldb.log lldb step")
        # self.runCmd("bt")

        # Check that the backtrace is what we expect:
        #  frame #0: 0x000000010d5e5f94 libTwo.dylib`tail_called_in_b_from_b at Two.c:7:3 [opt]
        #  frame #1: 0x000000010d5e5fa0 libTwo.dylib`tail_called_in_b_from_a [opt] [artificial]
        #  frame #2: 0x000000010d5dcf80 libOne.dylib`helper_in_a [opt] [artificial]
        #  frame #3: 0x000000010d5dcf79 libOne.dylib`tail_called_in_a_from_main at One.c:10:3 [opt]
        #  frame #4: 0x000000010d5d3f80 a.out`helper [opt] [artificial]
        #  frame #5: 0x000000010d5d3f79 a.out`main at main.c:10:3 [opt]
        expected_frames = [
                ("tail_called_in_b_from_b", False),
                ("tail_called_in_b_from_a", True),
                ("helper_in_a", True),
                ("tail_called_in_a_from_main", False),
                ("helper", True),
                ("main", False)
        ]
        for idx, (name, is_artificial) in enumerate(expected_frames):
            frame = thread.GetFrameAtIndex(idx)
            self.assertIn(name, frame.GetDisplayFunctionName())
            self.assertEqual(frame.IsArtificial(), is_artificial)
