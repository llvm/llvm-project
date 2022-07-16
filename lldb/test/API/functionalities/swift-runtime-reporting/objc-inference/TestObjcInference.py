# TestObjcInference.py
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
Test support of Swift Runtime Reporting for @objc inference.
"""
import lldb
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2
import json


class SwiftRuntimeReportingObjcInferenceTestCase(lldbtest.TestBase):

    @decorators.swiftTest
    @decorators.skipIfLinux
    def test_swift_runtime_reporting(self):
        self.build()
        self.do_test()

    def setUp(self):
        lldbtest.TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

        self.line_method = lldbtest.line_number(self.main_source, '// method line')
        self.line_method2 = lldbtest.line_number(self.main_source, '// method2 line')

    def do_test(self):
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        self.runCmd("run")

        self.expect("thread list",
                    substrs=['stopped', 'stop reason = implicit Objective-C entrypoint'])

        thread = target.process.GetSelectedThread()

        self.assertEqual(
            self.dbg.GetSelectedTarget().process.GetSelectedThread().GetStopReason(),
            lldb.eStopReasonInstrumentation)

        self.expect("thread info -s",
            substrs=["instrumentation_class", "issue_type", "description"])

        output_lines = self.res.GetOutput().split('\n')
        json_line = '\n'.join(output_lines[2:])
        data = json.loads(json_line)
        self.assertEqual(data["instrumentation_class"], "SwiftRuntimeReporting")
        self.assertEqual(data["issue_type"], "implicit-objc-entrypoint")
        self.assertEqual(data["description"],
            "implicit Objective-C entrypoint -[a.MyClass memberfunc] is deprecated and will be removed in Swift 4")
        self.assertEqual(len(data["notes"]), 1)
        self.assertEqual(data["notes"][0]["description"], "add '@objc' to expose this Swift declaration to Objective-C")
        self.assertEqual(len(data["notes"][0]["fixits"]), 1)
        self.assertTrue(data["notes"][0]["fixits"][0]["filename"].endswith(self.main_source))
        self.assertEqual(data["notes"][0]["fixits"][0]["start_line"], self.line_method)
        self.assertEqual(data["notes"][0]["fixits"][0]["end_line"], self.line_method)
        self.assertEqual(data["notes"][0]["fixits"][0]["start_col"], 3)
        self.assertEqual(data["notes"][0]["fixits"][0]["end_col"], 3)
        self.assertEqual(data["notes"][0]["fixits"][0]["replacement"], "@objc ")

        historical_threads = thread.GetStopReasonExtendedBacktraces(lldb.eInstrumentationRuntimeTypeSwiftRuntimeReporting)
        self.assertEqual(historical_threads.GetSize(), 1)

        self.runCmd("continue")

        self.expect("thread list",
                    substrs=['stopped', 'stop reason = implicit Objective-C entrypoint'])

        self.assertEqual(
            self.dbg.GetSelectedTarget().process.GetSelectedThread().GetStopReason(),
            lldb.eStopReasonInstrumentation)

        self.expect("thread info -s",
            substrs=["instrumentation_class", "issue_type", "description"])

        output_lines = self.res.GetOutput().split('\n')
        json_line = '\n'.join(output_lines[2:])
        data = json.loads(json_line)
        self.assertEqual(data["instrumentation_class"], "SwiftRuntimeReporting")
        self.assertEqual(data["issue_type"], "implicit-objc-entrypoint")
        self.assertEqual(data["description"],
            "implicit Objective-C entrypoint -[a.MyClass memberfunc2] is deprecated and will be removed in Swift 4")
        self.assertEqual(len(data["notes"]), 1)
        self.assertEqual(data["notes"][0]["description"], "add '@objc' to expose this Swift declaration to Objective-C")
        self.assertEqual(len(data["notes"][0]["fixits"]), 1)
        self.assertTrue(data["notes"][0]["fixits"][0]["filename"].endswith(self.main_source))
        self.assertEqual(data["notes"][0]["fixits"][0]["start_line"], self.line_method2)
        self.assertEqual(data["notes"][0]["fixits"][0]["end_line"], self.line_method2)
        self.assertEqual(data["notes"][0]["fixits"][0]["start_col"], 3)
        self.assertEqual(data["notes"][0]["fixits"][0]["end_col"], 3)
        self.assertEqual(data["notes"][0]["fixits"][0]["replacement"], "@objc ")

        historical_threads = thread.GetStopReasonExtendedBacktraces(lldb.eInstrumentationRuntimeTypeSwiftRuntimeReporting)
        self.assertEqual(historical_threads.GetSize(), 1)
