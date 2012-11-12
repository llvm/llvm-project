"""
Test that plugins that load commands work correctly.
"""

import os, time
import re
import unittest2
import lldb
from lldbtest import *
import lldbutil

class PluginCommandTestCase(TestBase):

    mydir = os.path.join("functionalities", "plugins", "commands")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_load_plugin(self):
        """Test that plugins that load commands work correctly."""

        # Invoke the default build rule.
        self.buildDefault()
        
        debugger = lldb.SBDebugger.Create()

        retobj = lldb.SBCommandReturnObject()
        
        retval = debugger.GetCommandInterpreter().HandleCommand("plugin load plugin.dylib",retobj)

        retobj.Clear()

        retval = debugger.GetCommandInterpreter().HandleCommand("plugin_loaded_command child abc def ghi",retobj)

        if self.TraceOn():
            print retobj.GetOutput()

        self.expect(retobj,substrs = ['abc def ghi'], exe=False)

        retobj.Clear()

        # check that abbreviations work correctly in plugin commands.
        retval = debugger.GetCommandInterpreter().HandleCommand("plugin_loaded_ ch abc def ghi",retobj)

        if self.TraceOn():
            print retobj.GetOutput()

        self.expect(retobj,substrs = ['abc def ghi'], exe=False)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
