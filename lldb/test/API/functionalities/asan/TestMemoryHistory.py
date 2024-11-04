"""
Test that ASan memory history provider returns correct stack traces
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbplatform
from lldbsuite.test import lldbutil


class AsanTestCase(TestBase):
    @skipIfFreeBSD  # llvm.org/pr21136 runtimes not yet available by default
    @expectedFailureNetBSD
    @skipUnlessAddressSanitizer
    def test(self):
        self.build()
        self.asan_tests()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.line_malloc = line_number("main.c", "// malloc line")
        self.line_malloc2 = line_number("main.c", "// malloc2 line")
        self.line_free = line_number("main.c", "// free line")
        self.line_breakpoint = line_number("main.c", "// break line")

    def asan_tests(self):
        target = self.createTestTarget()

        self.registerSanitizerLibrariesWithTarget(target)

        self.runCmd("breakpoint set -f main.c -l %d" % self.line_breakpoint)

        # "memory history" command should not work without a process
        self.expect(
            "memory history 0",
            error=True,
            substrs=["Command requires a current process"],
        )

        self.runCmd("run")

        stop_reason = (
            self.dbg.GetSelectedTarget().process.GetSelectedThread().GetStopReason()
        )
        if stop_reason == lldb.eStopReasonExec:
            # On OS X 10.10 and older, we need to re-exec to enable
            # interceptors.
            self.runCmd("continue")

        # the stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # test that the ASan dylib is present
        self.expect(
            "image lookup -n __asan_describe_address",
            "__asan_describe_address should be present",
            substrs=["1 match found"],
        )

        # test the 'memory history' command
        self.expect(
            "memory history 'pointer'",
            substrs=[
                "Memory deallocated by Thread",
                "a.out`f2",
                "main.c:%d" % self.line_free,
                "Memory allocated by Thread",
                "a.out`f1",
                "main.c:%d" % self.line_malloc,
            ],
        )

        # do the same using SB API
        process = self.dbg.GetSelectedTarget().process
        val = (
            process.GetSelectedThread().GetSelectedFrame().EvaluateExpression("pointer")
        )
        addr = val.GetValueAsUnsigned()
        threads = process.GetHistoryThreads(addr)
        self.assertEqual(threads.GetSize(), 2)

        history_thread = threads.GetThreadAtIndex(0)
        self.assertGreaterEqual(history_thread.num_frames, 2)
        self.assertEqual(
            history_thread.frames[1].GetLineEntry().GetFileSpec().GetFilename(),
            "main.c",
        )
        self.assertEqual(
            history_thread.frames[1].GetLineEntry().GetLine(), self.line_free
        )

        history_thread = threads.GetThreadAtIndex(1)
        self.assertGreaterEqual(history_thread.num_frames, 2)
        self.assertEqual(
            history_thread.frames[1].GetLineEntry().GetFileSpec().GetFilename(),
            "main.c",
        )
        self.assertEqual(
            history_thread.frames[1].GetLineEntry().GetLine(), self.line_malloc
        )

        # let's free the container (SBThreadCollection) and see if the
        # SBThreads still live
        threads = None
        self.assertGreaterEqual(history_thread.num_frames, 2)
        self.assertEqual(
            history_thread.frames[1].GetLineEntry().GetFileSpec().GetFilename(),
            "main.c",
        )
        self.assertEqual(
            history_thread.frames[1].GetLineEntry().GetLine(), self.line_malloc
        )

        # ASan will break when a report occurs and we'll try the API then
        self.runCmd("continue")

        self.expect(
            "thread list",
            "Process should be stopped due to ASan report",
            substrs=["stopped", "stop reason = Use of deallocated memory"],
        )

        # make sure the 'memory history' command still works even when we're
        # generating a report now
        self.expect(
            "memory history 'another_pointer'",
            substrs=[
                "Memory allocated by Thread",
                "a.out`f1",
                "main.c:%d" % self.line_malloc2,
            ],
        )
