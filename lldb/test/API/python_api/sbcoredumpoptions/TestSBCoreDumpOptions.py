"""Test the SBCoreDumpOptions APIs."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class SBCoreDumpOptionsAPICase(TestBase):
    def test_plugin_name_assignment(self):
        """Test assignment ensuring valid plugin names only."""
        options = lldb.SBCoreDumpOptions()
        error = options.SetPluginName(None)
        self.assertTrue(error.Success())
        self.assertEqual(options.GetPluginName(), None)
        error = options.SetPluginName("Not a real plugin")
        self.assertTrue(error.Success())
        self.assertEqual(options.GetPluginName(), None)
        error = options.SetPluginName("minidump")
        self.assertTrue(error.Success())
        self.assertEqual(options.GetPluginName(), "minidump")
        error = options.SetPluginName("")
        self.assertTrue(error.Success())
        self.assertEqual(options.GetPluginName(), None)

    def test_default_corestyle_behavior(self):
        """Test that the default core style is unspecified."""
        options = lldb.SBCoreDumpOptions()
        self.assertEqual(options.GetStyle(), lldb.eSaveCoreUnspecified)
