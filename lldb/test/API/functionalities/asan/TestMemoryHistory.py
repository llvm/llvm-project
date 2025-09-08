"""
Test that ASan memory history provider returns correct stack traces
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbplatform
from lldbsuite.test import lldbutil
from lldbsuite.test_event.build_exception import BuildError

class MemoryHistoryTestCase(TestBase):
    @skipIfFreeBSD  # llvm.org/pr21136 runtimes not yet available by default
    @expectedFailureNetBSD
    @skipUnlessAddressSanitizer
    def test_compiler_rt_asan(self):
        self.build(make_targets=["compiler_rt-asan"])
        self.compiler_rt_asan_tests()

    @skipUnlessDarwin
    @skipIf(bugnumber="rdar://109913184&143590169")
    def test_libsanitizers_asan(self):
        try:
            self.build(make_targets=["libsanitizers-asan"])
        except BuildError as e:
            self.skipTest("failed to build with libsanitizers")
        self.libsanitizers_asan_tests()

    @skipUnlessDarwin
    @skipIf(macos_version=["<", "15.5"])
    def test_libsanitizers_traces(self):
        self.build(make_targets=["libsanitizers-traces"])
        self.libsanitizers_traces_tests()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.line_malloc = line_number("main.c", "// malloc line")
        self.line_malloc2 = line_number("main.c", "// malloc2 line")
        self.line_free = line_number("main.c", "// free line")
        self.line_breakpoint = line_number("main.c", "// break line")

    def check_traces(self):
        self.expect(
            "memory history 'pointer'",
            substrs=[
                "Memory deallocated by Thread",
                "a.out`f2",
                f"main.c:{self.line_free}",
                "Memory allocated by Thread",
                "a.out`f1",
                f"main.c:{self.line_malloc}",
            ],
        )

    # Set breakpoint: after free, but before bug
    def set_breakpoint(self, target):
        bkpt = target.BreakpointCreateByLocation("main.c", self.line_breakpoint)
        self.assertGreater(bkpt.GetNumLocations(), 0, "Set the breakpoint successfully")

    def run_to_breakpoint(self, target):
        self.set_breakpoint(target)
        self.runCmd("run")
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

    def libsanitizers_traces_tests(self):
        target = self.createTestTarget()

        self.runCmd("env SanitizersAllocationTraces=all")

        self.run_to_breakpoint(target)
        self.check_traces()

    def libsanitizers_asan_tests(self):
        target = self.createTestTarget()

        self.runCmd("env SanitizersAddress=1 MallocSanitizerZone=1")

        self.run_to_breakpoint(target)
        self.check_traces()

        self.runCmd("continue")

        # Stop on report
        self.expect(
            "thread list",
            "Process should be stopped due to ASan report",
            substrs=["stopped", "stop reason = Use of deallocated memory"],
        )
        self.check_traces()

        # Make sure we're not stopped in the sanitizer library but instead at the
        # point of failure in the user-code.
        self.assertEqual(self.frame().GetFunctionName(), "main")

        # do the same using SB API
        process = self.dbg.GetSelectedTarget().process
        val = (
            process.GetSelectedThread().GetSelectedFrame().EvaluateExpression("pointer")
        )
        addr = val.GetValueAsUnsigned()
        threads = process.GetHistoryThreads(addr)
        self.assertEqual(threads.GetSize(), 2)

        history_thread = threads.GetThreadAtIndex(0)
        self.assertTrue(history_thread.num_frames >= 2)
        self.assertEqual(
            history_thread.frames[1].GetLineEntry().GetFileSpec().GetFilename(),
            "main.c",
        )

        history_thread = threads.GetThreadAtIndex(1)
        self.assertTrue(history_thread.num_frames >= 2)
        self.assertEqual(
            history_thread.frames[1].GetLineEntry().GetFileSpec().GetFilename(),
            "main.c",
        )

        # let's free the container (SBThreadCollection) and see if the
        # SBThreads still live
        threads = None
        self.assertTrue(history_thread.num_frames >= 2)
        self.assertEqual(
            history_thread.frames[1].GetLineEntry().GetFileSpec().GetFilename(),
            "main.c",
        )

    def compiler_rt_asan_tests(self):
        target = self.createTestTarget()

        self.registerSanitizerLibrariesWithTarget(target)

        self.set_breakpoint(target)

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

        self.check_traces()

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

        self.check_traces()

        # Make sure we're not stopped in the sanitizer library but instead at the
        # point of failure in the user-code.
        self.assertEqual(self.frame().GetFunctionName(), "main")

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
