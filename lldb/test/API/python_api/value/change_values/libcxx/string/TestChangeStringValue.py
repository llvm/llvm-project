"""
Test change libc++ string values.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class LibcxxChangeStringValueTestCase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def do_test_value(self, frame, var_name, new_value, str_prefix):
        str_value = frame.FindVariable(var_name)
        self.assertTrue(str_value.IsValid(), "Got the SBValue for {}".format(var_name))

        # update whole string
        err = lldb.SBError()
        result = str_value.SetValueFromCString(new_value, err)
        self.assertTrue(result, "Setting val returned error: {}".format(err))
        result = str_value.GetSummary() # str_value is a summary
        expected = '{}"{}"'.format(str_prefix, new_value)
        self.assertTrue(result == expected, "Got value: ({}), expected: ({})"
                                                .format(result, expected))

    @add_test_categories(["libc++"])
    @expectedFailureAll(oslist=["windows"], archs=["arm"], bugnumber="llvm.org/pr24772")
    @expectedFailureAll(archs=["arm"]) # arm can't jit
    def test(self):
        """Test that we can change values of libc++ string."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)
        bkpt = self.target().FindBreakpointByID(
            lldbutil.run_break_set_by_source_regexp(
                self, "Set break point at this line."))

        self.runCmd("run", RUN_SUCCEEDED)

        # Get Frame #0.
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition")
        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0.IsValid(), "Got a valid frame.")

        for var_name, str_prefix in zip(("s", "l", "ws", "wl", "u16s", "u32s"),
                                        ('',  '',  'L',  'L',  'u',    'U')):
            self.do_test_value(frame0, var_name, "new_value", str_prefix)
