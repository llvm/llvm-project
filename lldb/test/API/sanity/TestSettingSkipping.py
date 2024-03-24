"""
This is a sanity check that verifies that test can be skipped based on settings.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class SettingSkipSanityTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    CURRENT_PYTHON_VERSION = "3.0"

    @skipIf(py_version=(">=", CURRENT_PYTHON_VERSION))
    def testSkip(self):
        self.assertTrue(False, "This test should not run and fail (SKIPPED)")

    @skipIf(py_version=("<", CURRENT_PYTHON_VERSION))
    def testNoSKip(self):
        self.assertTrue(True, "This test should run and pass(PASS)")

    @expectedFailureAll(py_version=(">=", CURRENT_PYTHON_VERSION))
    def testXFAIL(self):
        self.assertTrue(False, "This test should expectedly fail (XFAIL)")

    @expectedFailureAll(py_version=("<", CURRENT_PYTHON_VERSION))
    def testNotXFAIL(self):
        self.assertTrue(True, "This test should pass (PASS)")

    @skipIf(setting=("target.i-made-this-one-up", "true"))
    def testNotExisting(self):
        self.assertTrue(True, "This test should run!")
