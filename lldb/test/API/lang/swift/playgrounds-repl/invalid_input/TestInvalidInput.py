# TestInvalidInput.py
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
import lldbsuite.test.lldbplaygroundrepl as repl
from lldbsuite.test.lldbtest import *


class TestInvalidInput(repl.PlaygroundREPLTest):

    def do_test(self):
        """
        Test that we can handle user correcting errors, submitting valid
        block 1, invalid block 2, then a revised version of block 2
        """

        # Execute first valid code
        result, output = self.execute_code("Input1.swift")
        playground_output = output.GetSummary()

        if self.is_compile_or_runtime_error(result):
            self.did_crash(result)
            self.assertTrue(False)

        self.assertTrue(playground_output is not None)
        self.assertTrue("pi=\\'3.14\\'" in playground_output)
        self.assertTrue("r=\\'2.0\\'" in playground_output)

        # Execute invalid block
        result, output = self.execute_code("Input2.swift")
        playground_output = output.GetSummary()

        # Make sure we got an appropriate error
        is_error = self.is_compile_or_runtime_error(result)
        self.assertTrue(is_error)
        error = self.get_stream_data(result)
        self.assertIn("left side of mutating operator", error, "Error messages do not match")
        self.assertIn(":15:3: error: left side of mutating operator", error, "Error line number does not match")

        # Execute revised block
        result, output = self.execute_code("Input3.swift")
        playground_output = output.GetSummary()

        if self.is_compile_or_runtime_error(result):
            self.did_crash(result)
            self.assertTrue(False)

        self.assertTrue(playground_output is not None)
        self.assertTrue("d=\\'4.0\\'" in playground_output)
        self.assertTrue("c=\\'12.56\\'" in playground_output)