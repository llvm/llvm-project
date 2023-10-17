"""
Test lldb data formatter callback-based matching.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class PythonSynthDataFormatterTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number("main.cpp", "// Set break point at this line.")

    def test_callback_matchers_api_registration(self):
        """Test data formatter commands."""
        self.build()

        _, process, thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec("main.cpp"), self.line
        )

        # Print derived without a formatter.
        self.expect("frame variable derived", substrs=["x = 2222", "y = 3333"])

        # now set up a summary function that uses a python callback to match
        # classes that derive from `Base`.
        self.runCmd(
            "command script import --allow-reload ./formatters_with_callback.py"
        )
        self.runCmd(
            "script formatters_with_callback.register_formatters(lldb.debugger)"
        )

        # Now `derived` should use our callback summary + synthetic children.
        self.expect(
            "frame variable derived",
            substrs=["hello from callback summary", "synthetic_child = 9999"],
        )

        # But not other classes.
        self.expect(
            "frame variable base",
            matching=False,
            substrs=["hello from callback summary"],
        )
        self.expect("frame variable base", substrs=["x = 1111"])

        self.expect(
            "frame variable nd", matching=False, substrs=["hello from callback summary"]
        )
        self.expect("frame variable nd", substrs=["z = 4444"])

    def test_callback_matchers_cli_registration(self):
        """Test data formatter commands."""
        self.build()

        _, process, thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec("main.cpp"), self.line
        )

        # Print derived without a formatter.
        self.expect("frame variable derived", substrs=["x = 2222", "y = 3333"])

        # now set up a summary function that uses a python callback to match
        # classes that derive from `Base`.
        self.runCmd(
            "command script import --allow-reload ./formatters_with_callback.py"
        )
        self.runCmd(
            "type summary add -e -s 'hello from callback summary' "
            "--recognizer-function formatters_with_callback.derives_from_base"
        )
        self.runCmd(
            "type synth add -l formatters_with_callback.SynthProvider "
            "--recognizer-function formatters_with_callback.derives_from_base"
        )

        # Now `derived` should use our callback summary + synthetic children.
        self.expect(
            "frame variable derived",
            substrs=["hello from callback summary", "synthetic_child = 9999"],
        )

        # But not other classes.
        self.expect(
            "frame variable base",
            matching=False,
            substrs=["hello from callback summary"],
        )
        self.expect("frame variable base", substrs=["x = 1111"])

        self.expect(
            "frame variable nd", matching=False, substrs=["hello from callback summary"]
        )
        self.expect("frame variable nd", substrs=["z = 4444"])
