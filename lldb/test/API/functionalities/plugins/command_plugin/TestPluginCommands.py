"""
Test that plugins that load commands work correctly.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class PluginCommandTestCase(TestBase):

    def setUp(self):
        TestBase.setUp(self)
        self.generateSource('plugin.cpp')

    @skipIfNoSBHeaders
    # Requires a compatible arch and platform to link against the host's built
    # lldb lib.
    @skipIfHostIncompatibleWithRemote
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    @no_debug_info_test
    def test_load_plugin(self):
        """Test that plugins that load commands work correctly."""

        plugin_name = "plugin"
        if sys.platform.startswith("darwin"):
            plugin_lib_name = "lib%s.dylib" % plugin_name
        else:
            plugin_lib_name = "lib%s.so" % plugin_name

        # Invoke the library build rule.
        self.buildLibrary("plugin.cpp", plugin_name)

        retobj = lldb.SBCommandReturnObject()

        retval = self.dbg.GetCommandInterpreter().HandleCommand(
            "plugin load %s" % self.getBuildArtifact(plugin_lib_name), retobj)

        retobj.Clear()

        retval = self.dbg.GetCommandInterpreter().HandleCommand(
            "plugin_loaded_command child abc def ghi", retobj)

        if self.TraceOn():
            print(retobj.GetOutput())

        self.expect(retobj, substrs=['abc def ghi'], exe=False)

        retobj.Clear()

        # check that abbreviations work correctly in plugin commands.
        retval = self.dbg.GetCommandInterpreter().HandleCommand(
            "plugin_loaded_ ch abc def ghi", retobj)

        if self.TraceOn():
            print(retobj.GetOutput())

        self.expect(retobj, substrs=['abc def ghi'], exe=False)

    @no_debug_info_test
    def test_invalid_plugin_invocation(self):
        self.expect("plugin load a b",
                    error=True, startstr="error: 'plugin load' requires one argument")
        self.expect("plugin load",
                    error=True, startstr="error: 'plugin load' requires one argument")

    @no_debug_info_test
    def test_invalid_plugin_target(self):
        self.expect("plugin load ThisIsNotAValidPluginName",
                    error=True, startstr="error: no such file")
