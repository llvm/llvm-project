"""Test the SBModuleSpec APIs."""

import lldb
from lldbsuite.test.lldbtest import TestBase


class TestSBModuleSpec(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        spec = lldb.SBModuleSpec()
        self.assertIsNone(spec.GetUUIDBytes())

        spec_uuid = "8FB5E28E344ECA77CE1969FD79A9B72AFD27C88F".encode("ascii")
        self.assertTrue(spec.SetUUIDBytes(spec_uuid))
        self.assertEqual(spec.GetUUIDBytes(), spec_uuid)
