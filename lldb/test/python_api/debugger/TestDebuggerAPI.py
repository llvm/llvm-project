"""
Test Debugger APIs.
"""

import os
import lldb
from lldbtest import TestBase, python_api_test


class DebuggerAPITestCase(TestBase):

    mydir = os.path.join("python_api", "debugger")

    @python_api_test
    def test_debugger_api_boundary_condition(self):
        """Exercise SBDebugger APIs with boundary conditions."""
        self.dbg.HandleCommand(None)
        self.dbg.SetDefaultArchitecture(None)
        self.dbg.GetScriptingLanguage(None)
        self.dbg.CreateTarget(None)
        self.dbg.CreateTarget(None, None, None, True, lldb.SBError())
        self.dbg.CreateTargetWithFileAndTargetTriple(None, None)
        self.dbg.CreateTargetWithFileAndArch(None, None)
        self.dbg.FindTargetWithFileAndArch(None, None)
        self.dbg.SetInternalVariable(None, None, None)
        self.dbg.GetInternalVariableValue(None, None)
        # FIXME (filcab): We must first allow for the swig bindings to know if
        # a Python callback is set. (Check python-typemaps.swig)
        #self.dbg.SetLoggingCallback(None)
        self.dbg.SetPrompt(None)
        self.dbg.SetCurrentPlatform(None)
        self.dbg.SetCurrentPlatformSDKRoot(None)

    @python_api_test
    def test_debugger_delete_invalid_target(self):
        """SBDebugger.DeleteTarget() should not crash LLDB given and invalid target."""
        target = lldb.SBTarget()
        self.assertFalse(target.IsValid())
        self.dbg.DeleteTarget(target)
