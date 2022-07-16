# TestExclusivityViolation.py
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
Test support of Swift Runtime Reporting for exclusivity violations.
"""
import lldb
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2
import json


class SwiftRuntimeReportingExclusivityViolationTestCase(lldbtest.TestBase):

    @decorators.swiftTest
    @decorators.skipIfLinux
    def test_swift_runtime_reporting(self):
        self.build()
        self.do_test()

    def setUp(self):
        lldbtest.TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

        self.line_breakpoint = lldbtest.line_number(self.main_source, '// get address line')
        self.line_current_access = lldbtest.line_number(self.main_source, '// current access line')
        self.line_previous_access = lldbtest.line_number(self.main_source, '// previous access line')

    def do_test(self):
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        self.runCmd("breakpoint set -f %s -l %d" % (self.main_source, self.line_breakpoint))

        self.runCmd("run")

        self.expect("thread list", lldbtest.STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', "stop reason = breakpoint"])

        thread = target.process.GetSelectedThread()
        frame = thread.GetSelectedFrame()

        frame_variables = frame.GetVariables(True, False, False, False)
        self.assertEquals(frame_variables.GetSize(), 1)
        self.assertEquals(frame_variables.GetValueAtIndex(0).GetName(), "p")

        addr = frame_variables.GetValueAtIndex(0).GetValueAsUnsigned()

        self.runCmd("continue")

        self.expect("thread list",
                    substrs=['stopped', 'stop reason = Simultaneous accesses'])

        self.assertEqual(
            self.dbg.GetSelectedTarget().process.GetSelectedThread().GetStopReason(),
            lldb.eStopReasonInstrumentation)

        self.expect("thread info -s",
            substrs=["instrumentation_class", "issue_type", "description"])

        output_lines = self.res.GetOutput().split('\n')
        json_line = '\n'.join(output_lines[2:])
        data = json.loads(json_line)
        self.assertEqual(data["instrumentation_class"], "SwiftRuntimeReporting")
        self.assertEqual(data["issue_type"], "exclusivity-violation")
        self.assertEqual(data["memory_address"], addr)
        self.assertEqual(data["description"],
            "Simultaneous accesses to 0x%lx, but modification requires exclusive access" % addr)

        historical_threads = thread.GetStopReasonExtendedBacktraces(lldb.eInstrumentationRuntimeTypeSwiftRuntimeReporting)
        self.assertEqual(historical_threads.GetSize(), 2)

        current_access = historical_threads.GetThreadAtIndex(0)
        found = False
        for i in range(0, current_access.GetNumFrames()):
            frame = current_access.GetFrameAtIndex(i)
            if frame.GetLineEntry().GetFileSpec().GetFilename() == self.main_source:
                if frame.GetLineEntry().GetLine() == self.line_current_access:
                    found = True
        self.assertTrue(found)

        previous_access = historical_threads.GetThreadAtIndex(1)
        self.assertEqual(previous_access.GetNumFrames(), 1)
        self.assertEqual(previous_access.GetFrameAtIndex(0).GetLineEntry().GetFileSpec().GetFilename(), self.main_source)
        self.assertEqual(previous_access.GetFrameAtIndex(0).GetLineEntry().GetLine(), self.line_previous_access)
