"""
This is a sanity check that verifies that `repr(sbobject)` and `str(sbobject)`
produce the same string.
"""


import lldb
from lldbsuite.test.lldbtest import *


class TestCase(TestBase):

  mydir = TestBase.compute_mydir(__file__)

  NO_DEBUG_INFO_TESTCASE = True

  def test(self):
    self.assertEqual(repr(self.dbg), str(self.dbg))
