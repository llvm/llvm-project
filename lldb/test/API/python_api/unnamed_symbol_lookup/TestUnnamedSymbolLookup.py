"""
Test lookup unnamed symbols.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

# --keep-symbol causes error on Windows: llvm-strip.exe: error: option is not supported for COFF
@skipIfWindows
class TestUnnamedSymbolLookup(TestBase):
    def test_unnamed_symbol_lookup(self):
        """Test looking up unnamed symbol synthetic name"""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_name_breakpoint(
            self, "main", exe_name="a.out.stripped"
        )

        main_frame = thread.GetFrameAtIndex(0)

        # Step until reaching the unnamed symbol called from main
        for _ in range(100):
            thread.StepInto()
            if thread.GetFrameAtIndex(0) != main_frame:
                break

            thread.StepInto()

        self.assertEqual(
            main_frame, thread.GetFrameAtIndex(1), "Expected to be called from main"
        )
        symbol = thread.GetFrameAtIndex(0).GetSymbol()
        self.assertIsNotNone(symbol, "unnamed symbol called from main not reached")
        self.assertTrue(symbol.name.startswith("___lldb_unnamed_symbol"))

        exe_module = symbol.GetStartAddress().GetModule()
        found_symbols = exe_module.FindSymbols(symbol.name)
        self.assertIsNotNone(found_symbols)
        self.assertEqual(found_symbols.GetSize(), 1)
