# TestAsanSwift.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Test Swift support of ASan.
"""
import lldb
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2
import json


class AsanSwiftTestCase(lldbtest.TestBase):

    @decorators.swiftTest
    @decorators.skipIfLinux
    @decorators.skipUnlessSwiftAddressSanitizer
    def test_asan_swift(self):
        self.build()
        self.do_test()

    def setUp(self):
        lldbtest.TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        self.line_breakpoint = lldbtest.line_number(
            self.main_source, '// breakpoint')

    def do_test(self):
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        runtimes = []
        for m in target.module_iter():
            libspec = m.GetFileSpec()
            if "clang_rt" in libspec.GetFilename():
                runtimes.append(os.path.join(libspec.GetDirectory(), libspec.GetFilename()))
        self.registerSharedLibrariesWithTarget(target, runtimes)

        self.runCmd(
            "breakpoint set -f %s -l %d" %
            (self.main_source, self.line_breakpoint))

        self.runCmd("run")

        stop_reason = self.dbg.GetSelectedTarget().process.GetSelectedThread().GetStopReason()
        if stop_reason == lldb.eStopReasonExec:
            # On OS X 10.10 and older, we need to re-exec to enable
            # interceptors.
            self.runCmd("continue")

        # the stop reason of the thread should be breakpoint.
        self.expect("thread list", lldbtest.STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        self.expect(
            "memory history `ptr`",
            substrs=[
                'Memory allocated by Thread 1',
                'main.swift'])

        # ASan will break when a report occurs and we'll try the API then
        self.runCmd("continue")

        # the stop reason of the thread should be a ASan report.
        self.expect("thread list", "Heap buffer overflow", substrs=[
                    'stopped', 'stop reason = Heap buffer overflow'])

        process = self.dbg.GetSelectedTarget().process
        thread = process.GetSelectedThread()

        self.assertEqual(
            thread.GetStopReason(),
            lldb.eStopReasonInstrumentation)

        for i in range(0, thread.GetNumFrames()):
            frame = thread.GetFrameAtIndex(i)
            if frame.GetFunctionName() == "main":
                self.expect("frame select %d" % i, substrs=["at main.swift"])
                break

        self.expect(
            "memory history `ptr`",
            substrs=[
                'Memory allocated by Thread 1',
                'main.swift'])
