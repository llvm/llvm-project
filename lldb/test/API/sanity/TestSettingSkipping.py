"""
This is a sanity check that verifies that test can be sklipped based on settings.
"""


import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class SettingSkipSanityTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(py_version=(">=", (3, 0)))
    def testSkip(self):
        """This setting is on by default"""
        self.assertTrue(False, "This test should not run!")

    @skipIf(py_version=("<", (3, 0)))
    def testNoMatch(self):
        self.assertTrue(True, "This test should run!")

    @skipIf(setting=("target.i-made-this-one-up", "true"))
    def testNotExisting(self):
        self.assertTrue(True, "This test should run!")

    @expectedFailureAll(py_version=(">=", (3, 0)))
    def testXFAIL(self):
        self.assertTrue(False, "This test should run and fail!")

    @expectedFailureAll(py_version=("<", (3, 0)))
    def testNotXFAIL(self):
        self.assertTrue(True, "This test should run and succeed!")
