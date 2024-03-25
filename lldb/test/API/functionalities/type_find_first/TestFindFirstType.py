"""
Test the SBModule and SBTarget type lookup APIs.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TypeFindFirstTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_find_first_type(self):
        """
        Test SBTarget::FindFirstType() and SBModule::FindFirstType() APIs.

        This function had regressed after some past modification of the type
        lookup internal code where if we had multiple types with the same
        basename, FindFirstType() could end up failing depending on which
        type was found first in the debug info indexes. This test will
        ensure this doesn't regress in the future.
        """
        self.build()
        target = self.createTestTarget()
        # Test the SBTarget APIs for FindFirstType
        integer_type = target.FindFirstType("Integer::Point")
        self.assertTrue(integer_type.IsValid())
        float_type = target.FindFirstType("Float::Point")
        self.assertTrue(float_type.IsValid())

        # Test the SBModule APIs for FindFirstType
        exe_module = target.GetModuleAtIndex(0)
        self.assertTrue(exe_module.IsValid())
        integer_type = exe_module.FindFirstType("Integer::Point")
        self.assertTrue(integer_type.IsValid())
        float_type = exe_module.FindFirstType("Float::Point")
        self.assertTrue(float_type.IsValid())
