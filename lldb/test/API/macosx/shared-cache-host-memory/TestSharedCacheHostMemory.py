import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SharedCacheHostMemoryTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    @skipIfRemote
    def test_host_lldb_memory(self):
        """Stop in a shared cache binary loaded from lldb's own memory and
        backtrace through it."""
        self.build()

        self.runCmd("settings set symbols.shared-cache-binary-loading host-lldb-memory")
        self.addTearDownHook(
            lambda: self.runCmd("settings clear symbols.shared-cache-binary-loading")
        )

        target = self.createTestTarget()

        # Breaking on puts forces lldb to load and symbolicate the shared cache
        # binary it lives in. The breakpoint resolves once that library is
        # loaded as the process launches.
        breakpoint = target.BreakpointCreateByName("puts")

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertState(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        self.assertGreater(
            breakpoint.GetNumLocations(),
            0,
            "puts breakpoint should resolve in the shared cache binary",
        )

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread, "Expected to stop at the puts breakpoint")

        # We should be able to walk the stack out of the shared cache binary
        # and back to main without crashing.
        self.assertEqual(thread.GetFrameAtIndex(0).GetFunctionName(), "puts")
        self.assertGreater(thread.GetNumFrames(), 1)
        frame_names = [
            thread.GetFrameAtIndex(i).GetFunctionName()
            for i in range(thread.GetNumFrames())
        ]
        self.assertIn("main", frame_names)
