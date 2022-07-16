"""
Test that an SBValue can update and format itself as its type changes
"""
import lldb
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbtest as lldbtest
import os
import unittest2


class TestSBValueUpdates(lldbtest.TestBase):

    @decorators.swiftTest
    def test_update_and_format_with_type_change(self):
        """Test that an SBValue can update and format itself as its type
        changes"""
        self.build()
        self.do_test()

    def setUp(self):
        lldbtest.TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Test that an SBValue can update and format itself as its type
        changes"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', self.main_source_spec)
        self.assertTrue(
            breakpoint.GetNumLocations() > 0, lldbtest.VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        self.runCmd("run")

        var_x = self.frame().FindVariable("x", lldb.eDynamicCanRunTarget)
        var_x.SetPreferSyntheticValue(True)

        self.assertTrue(var_x.GetValue() == "1")

        self.runCmd("continue")

        var_x = self.frame().FindVariable("x", lldb.eDynamicCanRunTarget)
        var_x.SetPreferSyntheticValue(True)

        self.assertTrue(var_x.GetSummary() == '"hi"')

        self.runCmd("continue")

        var_x = self.frame().FindVariable("x", lldb.eDynamicCanRunTarget)
        var_x.SetPreferSyntheticValue(True)

        self.assertTrue(var_x.GetValue() == "5")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
