"""
Test the lldb Python SBFormat API.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class FormatAPITestCase(TestBase):
    def test_format(self):
        format = lldb.SBFormat()
        self.assertFalse(format)

        error = lldb.SBError()
        format = lldb.SBFormat("${bad}", error)
        self.assertIn("invalid top level item 'bad'", error.GetCString())
        self.assertFalse(format)  # We expect an invalid object back if we have an error
        self.assertTrue(error.Fail())

        format = lldb.SBFormat("${frame.index}", error)
        self.assertIs(error.GetCString(), None)
        self.assertTrue(format)
        self.assertTrue(error.Success())
