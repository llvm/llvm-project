"""
Test that we embed the swig version into the lldb module
"""


"""
import os
import time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil
"""
from lldbsuite.test.lldbtest import *


class SwigVersionTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.assertTrue(getattr(lldb, "swig_version"))
        self.assertIsInstance(lldb.swig_version, tuple)
        self.assertEqual(len(lldb.swig_version), 3)
        self.assertGreaterEqual(lldb.swig_version[0], 4)
        for v in lldb.swig_version:
            self.assertGreaterEqual(v, 0)
