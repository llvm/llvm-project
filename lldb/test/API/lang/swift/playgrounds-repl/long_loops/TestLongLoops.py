# TestLongLoops.py
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
Test that long iteration loops don't crash
"""
import lldbsuite.test.lldbplaygroundrepl as repl
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestLongLoops(repl.PlaygroundREPLTest):

    @skipIfDarwin # This test is flakey
    def do_test(self):
        """
        Test that long iteration loops don't crash
        """

        # Execute for loop
        result, output = self.execute_code("ForLoop.swift")
        playground_output = output.GetSummary()

        # LLDB doesn't spit out all data for loops, check for crashes instead
        self.assertTrue(playground_output is not None)
        if self.is_compile_or_runtime_error(result):
            self.did_crash(result)
            self.assertTrue(False)

        # Execute nested for loop
        result, output = self.execute_code("NestedForLoop.swift")
        playground_output = output.GetSummary()

        self.assertTrue(playground_output is not None)
        if self.is_compile_or_runtime_error(result):
            self.did_crash(result)
            self.assertTrue(False)

        # Execute while loop
        result, output = self.execute_code("WhileLoop.swift")
        playground_output = output.GetSummary()

        self.assertTrue(playground_output is not None)
        if self.is_compile_or_runtime_error(result):
            self.did_crash(result)
            self.assertTrue(False)

        # Execute nested while loop
        result, output = self.execute_code("NestedWhileLoop.swift")
        playground_output = output.GetSummary()

        self.assertTrue(playground_output is not None)
        if self.is_compile_or_runtime_error(result):
            self.did_crash(result)
            self.assertTrue(False)
