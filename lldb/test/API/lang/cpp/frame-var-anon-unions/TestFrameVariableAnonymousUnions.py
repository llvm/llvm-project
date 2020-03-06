"""
Tests that frame variable looks into anonymous unions
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class FrameVariableAnonymousUnionsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_with_run_command(self):
        """Tests that frame variable looks into anonymous unions"""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        line = line_number('main.cpp', '// break here')
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", line, num_expected_locations=-1, loc_exact=False)

        self.runCmd("process launch", RUN_SUCCEEDED)

        process = self.dbg.GetSelectedTarget().GetProcess()

        if process.GetByteOrder() == lldb.eByteOrderLittle:
            self.expect('frame variable -f x i', substrs=['ffffff41'])
        else:
            self.expect('frame variable -f x i', substrs=['41ffff00'])

        self.expect('frame variable c', substrs=["'A"])

        self.expect('frame variable x', matching=False, substrs=['3'])
        self.expect('frame variable y', matching=False, substrs=["'B'"])
        self.expect('frame variable z', matching=False, substrs=['14'])
