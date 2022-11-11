import lldb
import os
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_full_path(self):
        file_spec = lldb.SBFileSpec()
        file_spec.SetDirectory("a")
        file_spec.SetFilename("b")
        self.assertEqual(file_spec.fullpath, os.path.join("a", "b"))
