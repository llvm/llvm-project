# TestNonREPLPlayground.py
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
Test that playgrounds work
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import unittest2

import sys
if sys.version_info.major == 2:
    import commands as subprocess
else:
    import subprocess

def execute_command(command):
    (exit_status, output) = subprocess.getstatusoutput(command)
    return exit_status


class TestNonREPLPlayground(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    @skipUnlessDarwin
    @swiftTest
    @skipIf(
        debug_info=decorators.no_match("dsym"),
        bugnumber="This test only builds one way")
    def test_playgrounds(self):
        """Test that playgrounds work"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here',
            lldb.SBFileSpec('PlaygroundStub.swift'),
            exe_name = "PlaygroundStub",
            extra_images=['libPlaygroundsRuntime.dylib'])

        contents = ""

        with open('Contents.swift', 'r') as contents_file:
            contents = contents_file.read()

        options = lldb.SBExpressionOptions()
        options.SetLanguage(lldb.eLanguageTypeSwift)
        options.SetPlaygroundTransformEnabled()

        self.frame().EvaluateExpression(contents, options)

        ret = self.frame().EvaluateExpression("get_output()")

        playground_output = ret.GetSummary()

        self.assertTrue(playground_output is not None)
        self.assertTrue("a=\\'3\\'" in playground_output)
        self.assertTrue("b=\\'5\\'" in playground_output)
        self.assertTrue("=\\'8\\'" in playground_output)
