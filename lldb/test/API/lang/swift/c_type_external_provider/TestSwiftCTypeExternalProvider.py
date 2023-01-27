"""
Tests that LLDB uses the external type info provider for C types.
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2



class TestSwiftCTypeExternalProvider(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    @swiftTest
    def test_swift_regex(self):
        """Test that C types with builtin metadata emitted are looked up using
        external type info provider."""
        self.build()
        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', self.main_source_spec)

        # Consult the second field to ensure we call GetIndexOfChildMemberWithName.
        self.expect('v dummy.second', substrs=['2'])

        # Make sure we look up the type with the external type info provider.
        provider_log_found = False
        with open(log, "r", encoding='utf-8') as logfile:
            for line in logfile:
                if '[LLDBTypeInfoProvider] Looking up debug type info for So5DummyV' in line:
                    provider_log_found = True
                    break
        self.assertTrue(provider_log_found)

