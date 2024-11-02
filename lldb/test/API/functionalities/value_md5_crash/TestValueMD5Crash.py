"""
Verify that the hash computing logic for ValueObject's values can't crash us.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ValueMD5CrashTestCase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// break here')

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24663")
    def test_with_run_command(self):
        """Verify that the hash computing logic for ValueObject's values can't crash us."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        value = self.frame().FindVariable("a")
        value.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)

        v = value.GetValue()
        type_name = value.GetTypeName()
        self.assertEquals(type_name, "B *", "a is a B*")

        self.runCmd("next")
        self.runCmd("process kill")

        # now the process is dead, and value needs updating
        v = value.GetValue()

        # if we are here, instead of crashed, the test succeeded
