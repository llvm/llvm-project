"""
Test Debugger APIs.
"""

import lldb

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DebuggerAPITestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @add_test_categories(['pyapi'])
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
        # self.dbg.SetLoggingCallback(None)
        self.dbg.SetPrompt(None)
        self.dbg.SetCurrentPlatform(None)
        self.dbg.SetCurrentPlatformSDKRoot(None)
        
        fresh_dbg = lldb.SBDebugger()
        self.assertEquals(len(fresh_dbg), 0)

    @add_test_categories(['pyapi'])
    def test_debugger_delete_invalid_target(self):
        """SBDebugger.DeleteTarget() should not crash LLDB given and invalid target."""
        target = lldb.SBTarget()
        self.assertFalse(target.IsValid())
        self.dbg.DeleteTarget(target)

    @add_test_categories(['pyapi'])
    def test_debugger_internal_variables(self):
        debugger_name = self.dbg.GetInstanceName()

        # Set a variable and check it was written successfully.
        error = lldb.SBDebugger.SetInternalVariable(
            'target.prefer-dynamic-value', 'no-dynamic-values', debugger_name)
        self.assertTrue(error.Success())
        ret = lldb.SBDebugger.GetInternalVariableValue(
            'target.prefer-dynamic-value', debugger_name)
        self.assertEqual(ret.GetSize(), 1)
        self.assertEqual(ret.GetStringAtIndex(0), 'no-dynamic-values')

        # Set a variable with a different value.
        error = lldb.SBDebugger.SetInternalVariable(
            'target.prefer-dynamic-value', 'no-run-target', debugger_name)
        self.assertTrue(error.Success())
        ret = lldb.SBDebugger.GetInternalVariableValue(
            'target.prefer-dynamic-value', debugger_name)
        self.assertEqual(ret.GetSize(), 1)
        self.assertEqual(ret.GetStringAtIndex(0), 'no-run-target')

        # Try setting invalid value, check for error.
        error = lldb.SBDebugger.SetInternalVariable(
            'target.prefer-dynamic-value', 'dummy-value', debugger_name)
        self.assertTrue(error.Fail())
        # Check that the value didn't change.
        ret = lldb.SBDebugger.GetInternalVariableValue(
            'target.prefer-dynamic-value', debugger_name)
        self.assertEqual(ret.GetSize(), 1)
        self.assertEqual(ret.GetStringAtIndex(0), 'no-run-target')
