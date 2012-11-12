"""Test evaluating expressions which ref. index variable 'i' which just goes
from out of scope to in scope when stopped at the breakpoint."""

import unittest2
import lldb
from lldbtest import *
import lldbutil

class NonOverlappingIndexVariableCase(TestBase):

    mydir = os.path.join("functionalities", "non-overlapping-index-variable-i")

    def setUp(self):
        TestBase.setUp(self)
        self.source = 'main.cpp'
        self.line_to_break = line_number(self.source, '// Set breakpoint here.')

    # rdar://problem/9890530
    def test_eval_index_variable(self):
        """Test expressions of variable 'i' which appears in two for loops."""
        self.buildDefault()
        self.exe_name = 'a.out'
        self.eval_index_variable_i(self.exe_name)

    def eval_index_variable_i(self, exe_name):
        """Test expressions of variable 'i' which appears in two for loops."""
        exe = os.path.join(os.getcwd(), exe_name)
        self.runCmd("file %s" % exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, self.source, self.line_to_break, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        self.runCmd('frame variable i')
        self.runCmd('expr i')
        self.runCmd('expr ptr[0]->point.x')
        self.runCmd('expr ptr[0]->point.y')
        self.runCmd('expr ptr[i]->point.x')
        self.runCmd('expr ptr[i]->point.y')

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
