"""Test the SBSaveCoreOptions APIs."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class SBSaveCoreOptionsAPICase(TestBase):
    def test_plugin_name_assignment(self):
        """Test assignment ensuring valid plugin names only."""
        options = lldb.SBSaveCoreOptions()
        error = options.SetPluginName(None)
        self.assertTrue(error.Success())
        self.assertEqual(options.GetPluginName(), None)
        error = options.SetPluginName("Not a real plugin")
        self.assertTrue(error.Fail())
        self.assertEqual(options.GetPluginName(), None)
        error = options.SetPluginName("minidump")
        self.assertTrue(error.Success())
        self.assertEqual(options.GetPluginName(), "minidump")
        error = options.SetPluginName("")
        self.assertTrue(error.Success())
        self.assertEqual(options.GetPluginName(), None)

    def test_default_corestyle_behavior(self):
        """Test that the default core style is unspecified."""
        options = lldb.SBSaveCoreOptions()
        self.assertEqual(options.GetStyle(), lldb.eSaveCoreUnspecified)

    def test_adding_and_removing_thread(self):
        """Test adding and removing a thread from save core options."""
        options = lldb.SBSaveCoreOptions()
        options.AddThread(1)
        removed_success = options.RemoveThreadID(1)
        self.assertTrue(removed_success)
        self.assertEqual(options.GetNumThreads(), 0)
        error = lldb.SBError()
        options.GetThreadAtIndex(0, error)
        self.assertTrue(error.Fail())
