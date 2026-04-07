"""
Test bt --provider * in a multithreaded scenario with even/odd thread-filtered
providers and an uppercase provider chained on top.
"""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test import lldbutil


class FrameProviderThreadFilterTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.source = "main.cpp"

    def build_and_stop_all_threads(self):
        """Build, set a breakpoint, and continue until all 3 worker threads
        have hit it. Returns (target, process, worker_threads)."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self,
            "break in thread",
            lldb.SBFileSpec(self.source),
            only_one_thread=False,
        )

        # The breakpoint is on a one-shot line (fetch_add), so each hit is
        # from a unique thread. Continue until all 3 have hit it.
        while bkpt.GetHitCount() < 3:
            process.Continue()

        # After 3 hits, all worker threads are alive: some in the spin loop,
        # one at the breakpoint. Collect all non-main threads.
        worker_threads = []
        for t in process:
            for i in range(t.GetNumFrames()):
                if "thread_work" in (t.GetFrameAtIndex(i).GetFunctionName() or ""):
                    worker_threads.append(t)
                    break

        self.assertEqual(len(worker_threads), 3, "Expected 3 worker threads")
        return target, process, worker_threads

    def register_providers(self, target):
        """Import the script and register all three providers in order."""
        script_path = os.path.join(self.getSourceDir(), "frame_provider.py")
        self.runCmd("command script import " + script_path)

        error = lldb.SBError()
        for cls in ("EvenThreadProvider", "OddThreadProvider", "UpperCaseProvider"):
            target.RegisterScriptedFrameProvider(
                "frame_provider." + cls,
                lldb.SBStructuredData(),
                error,
            )
            self.assertTrue(error.Success(), f"Should register {cls}: {error}")

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test_bt_provider_star_with_thread_filter(self):
        """
        Register EvenThreadProvider, OddThreadProvider, and UpperCaseProvider.
        For each worker thread, 'bt --provider *' should show:
        - Base Unwinder section with original names (thread_work).
        - UpperCaseProvider section with the fully chained result.
        - The non-applicable prefix provider should NOT appear.
        The default 'bt' should show the final upper-cased + prefixed names.
        """
        target, process, worker_threads = self.build_and_stop_all_threads()
        self.register_providers(target)

        for thread in worker_threads:
            prefix = "EVEN_THREAD_" if thread.GetIndexID() % 2 == 0 else "ODD_THREAD_"
            excluded = (
                "OddThreadProvider"
                if prefix == "EVEN_THREAD_"
                else "EvenThreadProvider"
            )

            process.SetSelectedThread(thread)

            self.expect(
                "bt --provider '*'",
                substrs=["Base Unwinder", "thread_work", "UpperCaseProvider", prefix],
            )
            self.expect(
                "bt --provider '*'",
                matching=False,
                substrs=[excluded],
            )
            self.expect("bt", substrs=[prefix])
