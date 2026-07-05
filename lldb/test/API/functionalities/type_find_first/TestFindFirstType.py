"""
Test the SBModule and SBTarget type lookup APIs.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TypeFindFirstTestCase(TestBase):
    def test_find_first_type(self):
        """
        Test SBTarget::FindFirstType() and SBModule::FindFirstType() APIs.

        This function had regressed after some past modification of the type
        lookup internal code where if we had multiple types with the same
        basename, FindFirstType() could end up failing depending on which
        type was found first in the debug info indexes. This test will
        ensure this doesn't regress in the future.

        The test also looks for a type defined in a different compilation unit
        to verify that SymbolFileDWARFDebugMap searches each symbol file in a
        module.
        """
        self.build()
        target = self.createTestTarget()
        exe_module = target.GetModuleAtIndex(0)
        self.assertTrue(exe_module.IsValid())
        # Test the SBTarget and SBModule APIs for FindFirstType
        for api in [target, exe_module]:
            integer_type = api.FindFirstType("Integer::Point")
            self.assertTrue(integer_type.IsValid())
            float_type = api.FindFirstType("Float::Point")
            self.assertTrue(float_type.IsValid())
            external_type = api.FindFirstType("OtherCompilationUnit::Type")
            self.assertTrue(external_type.IsValid())
            nonexistent_type = api.FindFirstType("NonexistentType")
            self.assertFalse(nonexistent_type.IsValid())
