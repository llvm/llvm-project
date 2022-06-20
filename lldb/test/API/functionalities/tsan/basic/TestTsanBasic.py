"""
Tests basic ThreadSanitizer support (detecting a data race).
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import json


class TsanBasicTestCase(TestBase):

    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="non-core functionality, need to reenable and fix later (DES 2014.11.07)")
    @expectedFailureNetBSD
    @skipIfFreeBSD  # llvm.org/pr21136 runtimes not yet available by default
    @skipIfRemote
    @skipUnlessThreadSanitizer
    @no_debug_info_test
    def test(self):
        self.build()
        self.tsan_tests()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.line_malloc = line_number('main.c', '// malloc line')
        self.line_thread1 = line_number('main.c', '// thread1 line')
        self.line_thread2 = line_number('main.c', '// thread2 line')

    def tsan_tests(self):
        exe = self.getBuildArtifact("a.out")
        self.expect(
            "file " + exe,
            patterns=["Current executable set to .*a.out"])

        self.runCmd("run")

        stop_reason = self.dbg.GetSelectedTarget().process.GetSelectedThread().GetStopReason()
        if stop_reason == lldb.eStopReasonExec:
            # On OS X 10.10 and older, we need to re-exec to enable
            # interceptors.
            self.runCmd("continue")

        # the stop reason of the thread should be breakpoint.
        self.expect("thread list", "A data race should be detected",
                    substrs=['stopped', 'stop reason = Data race detected'])

        self.assertEqual(
            self.dbg.GetSelectedTarget().process.GetSelectedThread().GetStopReason(),
            lldb.eStopReasonInstrumentation)

        # test that the TSan dylib is present
        self.expect(
            "image lookup -n __tsan_get_current_report",
            "__tsan_get_current_report should be present",
            substrs=['1 match found'])

        # We should be stopped in __tsan_on_report
        process = self.dbg.GetSelectedTarget().process
        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()
        self.assertIn("__tsan_on_report", frame.GetFunctionName())

        # The stopped thread backtrace should contain either line1 or line2
        # from main.c.
        found = False
        for i in range(0, thread.GetNumFrames()):
            frame = thread.GetFrameAtIndex(i)
            if frame.GetLineEntry().GetFileSpec().GetFilename() == "main.c":
                if frame.GetLineEntry().GetLine() == self.line_thread1:
                    found = True
                if frame.GetLineEntry().GetLine() == self.line_thread2:
                    found = True
        self.assertTrue(found)

        self.expect(
            "thread info -s",
            "The extended stop info should contain the TSan provided fields",
            substrs=[
                "instrumentation_class",
                "description",
                "mops"])

        output_lines = self.res.GetOutput().split('\n')
        json_line = '\n'.join(output_lines[2:])
        data = json.loads(json_line)
        self.assertEqual(data["instrumentation_class"], "ThreadSanitizer")
        self.assertEqual(data["issue_type"], "data-race")
        self.assertEqual(len(data["mops"]), 2)

        backtraces = thread.GetStopReasonExtendedBacktraces(
            lldb.eInstrumentationRuntimeTypeAddressSanitizer)
        self.assertEqual(backtraces.GetSize(), 0)

        backtraces = thread.GetStopReasonExtendedBacktraces(
            lldb.eInstrumentationRuntimeTypeThreadSanitizer)
        self.assertTrue(backtraces.GetSize() >= 2)

        # First backtrace is a memory operation
        thread = backtraces.GetThreadAtIndex(0)
        found = False
        for i in range(0, thread.GetNumFrames()):
            frame = thread.GetFrameAtIndex(i)
            if frame.GetLineEntry().GetFileSpec().GetFilename() == "main.c":
                if frame.GetLineEntry().GetLine() == self.line_thread1:
                    found = True
                if frame.GetLineEntry().GetLine() == self.line_thread2:
                    found = True
        self.assertTrue(found)

        # Second backtrace is a memory operation
        thread = backtraces.GetThreadAtIndex(1)
        found = False
        for i in range(0, thread.GetNumFrames()):
            frame = thread.GetFrameAtIndex(i)
            if frame.GetLineEntry().GetFileSpec().GetFilename() == "main.c":
                if frame.GetLineEntry().GetLine() == self.line_thread1:
                    found = True
                if frame.GetLineEntry().GetLine() == self.line_thread2:
                    found = True
        self.assertTrue(found)

        self.runCmd("continue")

        # the stop reason of the thread should be a SIGABRT.
        self.expect("thread list", "We should be stopped due a SIGABRT",
                    substrs=['stopped', 'stop reason = signal SIGABRT'])

        # test that we're in pthread_kill now (TSan abort the process)
        self.expect("thread list", "We should be stopped in pthread_kill",
                    substrs=['pthread_kill'])
