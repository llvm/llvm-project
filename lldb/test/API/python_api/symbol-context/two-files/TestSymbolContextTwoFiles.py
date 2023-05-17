"""
Test SBSymbolContext APIs.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SymbolContextTwoFilesTestCase(TestBase):

    @expectedFailureAll(oslist=["windows"])
    def test_lookup_by_address(self):
        """Test lookup by address in a module with multiple compilation units"""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        module = target.GetModuleAtIndex(0)
        self.assertTrue(module.IsValid())
        for symbol_name in ["struct1::f()", "struct2::f()"]:
            sc_list = module.FindFunctions(symbol_name, lldb.eSymbolTypeCode)
            self.assertTrue(1, sc_list.GetSize())
            symbol_address = sc_list.GetContextAtIndex(
                0).GetSymbol().GetStartAddress()
            self.assertTrue(symbol_address.IsValid())
            sc_by_address = module.ResolveSymbolContextForAddress(
                symbol_address, lldb.eSymbolContextFunction)
            self.assertEqual(symbol_name,
                             sc_by_address.GetFunction().GetName())

    def test_ranges_in_multiple_compile_unit(self):
        """This test verifies that we correctly handle the case when multiple
        compile unit contains DW_AT_ranges and DW_AT_ranges_base attributes."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        source1 = "file1.cpp"
        line1 = line_number(source1, '// Break1')
        breakpoint1 = target.BreakpointCreateByLocation(source1, line1)
        self.assertIsNotNone(breakpoint1)
        self.assertTrue(breakpoint1.IsValid())

        source2 = "file2.cpp"
        line2 = line_number(source2, '// Break2')
        breakpoint2 = target.BreakpointCreateByLocation(source2, line2)
        self.assertIsNotNone(breakpoint2)
        self.assertTrue(breakpoint2.IsValid())

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertIsNotNone(process, PROCESS_IS_VALID)

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint2)
        self.assertEqual(len(threads), 1)
        frame = threads[0].GetFrameAtIndex(0)
        value = frame.FindVariable("x")
        self.assertTrue(value.IsValid())

        process.Continue()

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint1)
        self.assertEqual(len(threads), 1)
        frame = threads[0].GetFrameAtIndex(0)
        value = frame.FindVariable("x")
        self.assertTrue(value.IsValid())

        process.Continue()
