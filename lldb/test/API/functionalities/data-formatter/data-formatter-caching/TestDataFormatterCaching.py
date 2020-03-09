import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestDataFormatterCaching(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_with_run_command(self):
        """
        Test that hardcoded summary formatter matches aren't improperly cached.
        """
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('a.c'))
        valobj = self.frame().FindVariable('f')
        self.assertEqual(valobj.GetValue(), '4')
        bkpt_b = target.BreakpointCreateBySourceRegex('break here',
                                                      lldb.SBFileSpec('b.c'))
        lldbutil.continue_to_breakpoint(process, bkpt_b)
        valobj = self.frame().FindVariable('f4')
        self.assertEqual(valobj.GetSummary(), '(1, 2, 3, 4)')
