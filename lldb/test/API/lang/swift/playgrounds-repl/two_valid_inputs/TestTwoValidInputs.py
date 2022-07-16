# TestTwoValidBlocks.py
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
Test that statements made in one block can be referenced in a proceeding block
"""
from __future__ import print_function
import lldbsuite.test.lldbplaygroundrepl as repl
from lldbsuite.test.lldbtest import *

class TestTwoValidInputs(repl.PlaygroundREPLTest):

    def do_test(self):
        """
        Test that statements made in one block can be referenced in a
        proceeding block
        """

        # Execute first block
        result, output = self.execute_code("Input1.swift")
        playground_output = output.GetSummary()

        if self.is_compile_or_runtime_error(result):
            self.did_crash(result)
            self.assertTrue(False)

        with recording(self, self.TraceOn()) as sbuf:
            print("playground output:", file=sbuf)
            print(playground_output, file=sbuf)

        self.assertTrue(playground_output is not None)
        self.assertTrue("a=\\'3\\'" in playground_output)
        self.assertTrue("b=\\'5\\'" in playground_output)

        # Execute second block
        result, output = self.execute_code("Input2.swift")
        playground_output = output.GetSummary()

        if self.is_compile_or_runtime_error(result):
            self.did_crash(result)
            self.assertTrue(False)

        self.assertTrue(playground_output is not None)
        self.assertTrue("=\\'8\\'" in playground_output)