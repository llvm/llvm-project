"""
Test loading of a kext binary.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LoadKextTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        #super(LoadKextTestCase, self).setUp()
        #self._initial_platform = lldb.DBG.GetSelectedPlatform()

    def test_load_kext(self):
        """Test that lldb can load a kext binary."""

        # Create kext from YAML.
        self.yaml2obj("mykext.yaml", self.getBuildArtifact("mykext"))

        target = self.dbg.CreateTarget(self.getBuildArtifact("mykext"))

        self.assertTrue(target.IsValid())

        self.assertEqual(target.GetNumModules(), 1)
        mod = target.GetModuleAtIndex(0)
        self.assertEqual(mod.GetFileSpec().GetFilename(), "mykext")
