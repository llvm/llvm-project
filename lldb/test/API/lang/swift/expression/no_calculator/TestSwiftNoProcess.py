# TestSwiftNoProcess.py
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
Tests simple swift expressions
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftNoProcess(TestBase):

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    @swiftTest
    @skipIf(oslist=['linux', 'windows'])
    def test_swift_no_target(self):
        """Tests that we give a reasonable error if we try to run expressions with no target"""
        result = lldb.SBCommandReturnObject()
        ret_val = self.dbg.GetCommandInterpreter().HandleCommand(
            "expression -l swift -- 1 + 2", result)
        self.assertTrue(
            ret_val == lldb.eReturnStatusFailed,
            "Swift expression with no target should fail.")

    @swiftTest
    @skipIf(oslist=['windows'])
    def test_swift_no_process(self):
        """Tests that we give a reasonable error if we try to run expressions with no process"""
        self.build()

        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        result = lldb.SBCommandReturnObject()
        options = lldb.SBExpressionOptions()
        options.SetLanguage(lldb.eLanguageTypeSwift)
        ret_val = target.EvaluateExpression("1 + 2", options)
        self.assertTrue(ret_val.GetError().Fail(),
                        "Swift expressions with no process should fail.")


