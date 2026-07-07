"""
Test SBFileSpec APIs, with emphasis on equality comparisons against strings.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class FileSpecAPITestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_filespec_eq(self):
        """Test SBFileSpec equality comparisons."""
        empty = lldb.SBFileSpec()
        self.assertTrue(empty == lldb.SBFileSpec())
        self.assertTrue(empty == "")
        self.assertTrue(empty == lldb.SBFileSpec(""))
        self.assertFalse(empty != "")
        self.assertTrue(not empty)
        self.assertFalse(empty is None)

    def test_filespec_eq_path(self):
        """Test SBFileSpec equality with non-empty path strings."""
        spec = lldb.SBFileSpec("/a/b")
        self.assertTrue(spec == "/a/b")
        self.assertFalse(spec == "/a/c")
        self.assertFalse(spec != "/a/b")
        self.assertTrue(spec != "/a/c")

    def test_filespec_eq_other_type(self):
        """Test SBFileSpec equality with unsupported types returns False."""
        spec = lldb.SBFileSpec()
        self.assertFalse(spec == 42)
        self.assertFalse(spec == [])
