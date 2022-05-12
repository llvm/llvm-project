"""
Test SBCompileUnit APIs.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CompileUnitAPITestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    def test(self):
        """Exercise some SBCompileUnit APIs."""
        self.build()

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self, 'break here', lldb.SBFileSpec('main.c'))
        self.assertTrue(target, VALID_TARGET)
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertTrue(bkpt and bkpt.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition")
        frame0 = thread.GetFrameAtIndex(0)
        line_entry = frame0.GetLineEntry()

        sc_list = target.FindCompileUnits(line_entry.GetFileSpec())
        self.assertGreater(sc_list.GetSize(), 0)

        main_cu = sc_list.compile_units[0]
        self.assertTrue(main_cu.IsValid(), "Main executable CU is not valid")

        self.assertEqual(main_cu.GetIndexForLineEntry(line_entry, True),
                         main_cu.FindLineEntryIndex(0, line_entry.GetLine(),
                                   line_entry.GetFileSpec(), True))


