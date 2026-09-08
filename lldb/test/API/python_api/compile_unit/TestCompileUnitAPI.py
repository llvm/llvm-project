"""
Test SBCompileUnit APIs.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CompileUnitAPITestCase(TestBase):
    def test(self):
        """Exercise some SBCompileUnit APIs."""
        self.build()

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )
        self.assertTrue(target, VALID_TARGET)
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertTrue(bkpt and bkpt.GetNumLocations() == 1, VALID_BREAKPOINT)

        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition",
        )
        frame0 = thread.GetFrameAtIndex(0)
        line_entry = frame0.GetLineEntry()

        sc_list = target.FindCompileUnits(line_entry.GetFileSpec())
        self.assertGreater(sc_list.GetSize(), 0)

        main_cu = sc_list.compile_units[0]
        self.assertTrue(main_cu.IsValid(), "Main executable CU is not valid")

        a_mod: lldb.SBModule = target.FindModule(lldb.SBFileSpec("a.out"))
        main_cu_by_index = a_mod.compile_unit[0]
        self.assertTrue(main_cu_by_index.IsValid(), "Main executable CU is not valid")

        main_cu_by_name = a_mod.compile_unit["main.c"]
        self.assertTrue(main_cu_by_name.IsValid(), "Main executable CU is not valid")

        main_cu_by_regex_list = a_mod.compile_unit[re.compile(r".*main.*")]
        self.assertEqual(len(main_cu_by_regex_list), 1)
        [main_cu_by_regex] = main_cu_by_regex_list
        self.assertTrue(main_cu_by_regex.IsValid(), "Main executable CU is not valid")

        self.assertEqual(
            main_cu.FindLineEntryIndex(line_entry, True),
            main_cu.FindLineEntryIndex(
                0, line_entry.GetLine(), line_entry.GetFileSpec(), True
            ),
        )
