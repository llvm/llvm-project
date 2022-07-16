# TestSwiftRuntimeFailureRecognizer.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2020 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Test Swift Runtime Failure Recognizer
"""
import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftRuntimeRecognizer(lldbtest.TestBase):

    @swiftTest
    def test_swift_runtime_recognizer(self):
        """Test Swift Runtime Failure Recognizer"""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"))
        self.runCmd("process launch")

        self.expect("frame recognizer list",
                    substrs=['Swift Runtime Failure StackFrame Recognizer, symbol Swift runtime failure (regexp)'])


        self.expect("frame recognizer info 0",
                    substrs=['frame 0 is recognized by Swift Runtime Failure StackFrame Recognizer'])

        self.expect("thread info",
                    substrs=['stop reason = Swift runtime failure: arithmetic overflow'])

        self.expect("frame info",
                    patterns=['frame #1(.*)`testit(.*)at RuntimeFailure\.swift'])
