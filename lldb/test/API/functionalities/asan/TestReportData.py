"""
Test the AddressSanitizer runtime support for report breakpoint and data extraction.
"""

import json
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test_event.build_exception import BuildError


class AsanTestReportDataCase(TestBase):
    @skipIfFreeBSD  # llvm.org/pr21136 runtimes not yet available by default
    @expectedFailureNetBSD
    @skipUnlessAddressSanitizer
    @skipIf(archs=["i386"], bugnumber="llvm.org/PR36710")
    def test(self):
        self.build(make_targets=["compiler_rt-asan"])
        self.asan_tests()

    @skipUnlessDarwin
    @skipIf(bugnumber="rdar://109913184&143590169")
    def test_libsanitizers_asan(self):
        try:
            self.build(make_targets=["libsanitizers-asan"])
        except BuildError as e:
            self.skipTest("failed to build with libsanitizers")
        self.asan_tests(libsanitizers=True)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.line_malloc = line_number("main.c", "// malloc line")
        self.line_malloc2 = line_number("main.c", "// malloc2 line")
        self.line_free = line_number("main.c", "// free line")
        self.line_breakpoint = line_number("main.c", "// break line")
        self.line_crash = line_number("main.c", "// BOOM line")
        self.col_crash = 16

    def asan_tests(self, libsanitizers=False):
        target = self.createTestTarget()

        if libsanitizers:
            self.runCmd("env SanitizersAddress=1 MallocSanitizerZone=1")
        else:
            self.registerSanitizerLibrariesWithTarget(target)

        self.runCmd("run")

        stop_reason = (
            self.dbg.GetSelectedTarget().process.GetSelectedThread().GetStopReason()
        )
        if stop_reason == lldb.eStopReasonExec:
            # On OS X 10.10 and older, we need to re-exec to enable
            # interceptors.
            self.runCmd("continue")

        self.expect(
            "thread list",
            "Process should be stopped due to ASan report",
            substrs=["stopped", "stop reason = Use of deallocated memory"],
        )

        self.assertEqual(
            self.dbg.GetSelectedTarget().process.GetSelectedThread().GetStopReason(),
            lldb.eStopReasonInstrumentation,
        )

        if self.platformIsDarwin():
            # Make sure we're not stopped in the sanitizer library but instead at the
            # point of failure in the user-code.
            self.assertEqual(self.frame().GetFunctionName(), "main")

        self.expect(
            "bt",
            "The backtrace should show the crashing line",
            substrs=["main.c:%d:%d" % (self.line_crash, self.col_crash)],
        )

        self.expect(
            "thread info -s",
            "The extended stop info should contain the ASan provided fields",
            substrs=[
                "access_size",
                "access_type",
                "address",
                "description",
                "heap-use-after-free",
                "pc",
            ],
        )

        output_lines = self.res.GetOutput().split("\n")
        json_line = "\n".join(output_lines[2:])
        data = json.loads(json_line)
        self.assertEqual(data["description"], "heap-use-after-free")
        self.assertEqual(data["instrumentation_class"], "AddressSanitizer")
        self.assertEqual(data["stop_type"], "fatal_error")

        # now let's try the SB API
        process = self.dbg.GetSelectedTarget().process
        thread = process.GetSelectedThread()

        s = lldb.SBStream()
        self.assertTrue(thread.GetStopReasonExtendedInfoAsJSON(s))
        s = s.GetData()
        data2 = json.loads(s)
        self.assertEqual(data, data2)
