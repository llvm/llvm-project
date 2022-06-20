"""
This is a sanity check that verifies that test can be sklipped based on settings.
"""


import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class SettingSkipSanityTestCase(TestBase):

  NO_DEBUG_INFO_TESTCASE = True

  @skipIf(setting=('target.prefer-dynamic-value', 'no-dynamic-values'))
  def testSkip(self):
    """This setting is on by default"""
    self.assertTrue(False, "This test should not run!")

  @skipIf(setting=('target.prefer-dynamic-value', 'run-target'))
  def testNoMatch(self):
    self.assertTrue(True, "This test should run!")

  @skipIf(setting=('target.i-made-this-one-up', 'true'))
  def testNotExisting(self):
    self.assertTrue(True, "This test should run!")

  @expectedFailureAll(setting=('target.prefer-dynamic-value', 'no-dynamic-values'))
  def testXFAIL(self):
    self.assertTrue(False, "This test should run and fail!")

  @expectedFailureAll(setting=('target.prefer-dynamic-value', 'run-target'))
  def testNotXFAIL(self):
    self.assertTrue(True, "This test should run and succeed!")

