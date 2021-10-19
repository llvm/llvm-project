import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_FindTypes_on_scratch_AST(self):
        """
        Tests FindTypes invoked with only LLDB's scratch AST present.
        """
        target = self.dbg.GetDummyTarget()
        # There should be only one instance of 'unsigned long' in our single
        # scratch AST. Note: FindTypes/SBType hahave no filter by language, so
        # pick something that is unlikely to also be found in the scratch
        # TypeSystem of other language plugins.
        self.assertEqual(len(target.FindTypes("unsigned long")), 1)
