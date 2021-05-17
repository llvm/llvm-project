"""Test SBValue::GetValueDidChange"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ValueVarUpdateTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_with_process_launch_api(self):
        """Test SBValue::GetValueDidChange"""
        # Get the full path to our executable to be attached/debugged.
        exe = self.getBuildArtifact(self.testMethodName)
        d = {'EXE': exe}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        target = self.dbg.CreateTarget(exe)

        breakpoint = target.BreakpointCreateBySourceRegex(
            "break here", lldb.SBFileSpec("main.c"))

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        i = self.frame().FindVariable("i")
        i_val = i.GetValueAsUnsigned(0)
        c = self.frame().FindVariable("c")

        # Update any values from the SBValue objects so we can ask them if they
        # changed after a continue
        i.GetValueDidChange()
        c.GetChildAtIndex(1).GetValueDidChange()
        c.GetChildAtIndex(0).GetChildAtIndex(0).GetValueDidChange()

        if self.TraceOn():
            self.runCmd("frame variable")

        self.runCmd("continue")

        if self.TraceOn():
            self.runCmd("frame variable")

        self.assertTrue(
            i_val != i.GetValueAsUnsigned(0),
            "GetValue() is saying a lie")
        self.assertTrue(
            i.GetValueDidChange(),
            "GetValueDidChange() is saying a lie")

        # Check complex type
        self.assertTrue(c.GetChildAtIndex(0).GetChildAtIndex(0).GetValueDidChange(
        ) and not c.GetChildAtIndex(1).GetValueDidChange(), "GetValueDidChange() is saying a lie")
