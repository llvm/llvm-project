"""Test SBTarget::FindSymbolContexts / SBModule::FindSymbolContexts APIs."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class FindSymbolContextsAPITestCase(TestBase):
    def test_finds_inlined_header_entry(self):
        """
        `FindSymbolContexts` should return the inlined instance of a
        header-defined function even though the header is not the caller's
        compile unit. The pre-existing `SBCompileUnit::GetLineEntryAtIndex`
        walk cannot see those entries.
        """
        self.build()
        target, *_ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        self.assertTrue(target, VALID_TARGET)

        header_spec = lldb.SBFileSpec("inlined.h")
        inlined_line = line_number("inlined.h", "// inlined body")

        line_spec = lldb.SBLineSpec(header_spec, inlined_line)
        sc_list: lldb.SBSymbolContextList = target.FindSymbolContexts(line_spec)
        self.assertGreater(sc_list.GetSize(), 0)

        for sc in sc_list:
            self.assertTrue(sc.GetModule().IsValid())
            self.assertTrue(sc.GetCompileUnit().IsValid())
            self.assertTrue(sc.GetFunction().IsValid())
            entry = sc.GetLineEntry()
            self.assertTrue(entry.IsValid())
            self.assertEqual(entry.GetLine(), inlined_line)
            self.assertEqual(entry.GetFileSpec().GetFilename(), "inlined.h")

    def test_check_inlines_false(self):
        """check_inlines=False excludes header inlines when the header is not a primary compile unit."""
        self.build()
        target, *_ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        header_spec = lldb.SBFileSpec("inlined.h")
        inlined_line = line_number("inlined.h", "// inlined body")

        line_spec = lldb.SBLineSpec(header_spec, inlined_line)
        line_spec.SetCheckInlines(False)
        sc_list = target.FindSymbolContexts(line_spec)
        self.assertEqual(sc_list.GetSize(), 0)

    def test_primary_source(self):
        """Both check_inlines values return matches for a primary-CU line."""
        self.build()
        target, *_ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        main_spec = lldb.SBFileSpec("main.cpp")
        break_line = line_number("main.cpp", "// break here")

        for check_inlines in (True, False):
            line_spec = lldb.SBLineSpec(main_spec, break_line)
            line_spec.SetCheckInlines(check_inlines)
            sc_list = target.FindSymbolContexts(line_spec)
            self.assertGreater(sc_list.GetSize(), 0, f"check_inlines={check_inlines}")

    def test_matches_between_target_and_module(self):
        """SBTarget and SBModule return the same matches for a single-module executable."""
        self.build()
        target, *_ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        main_spec = lldb.SBFileSpec("main.cpp")
        break_line = line_number("main.cpp", "// break here")
        source_location = lldb.SBLineSpec(main_spec, break_line)

        target_list = target.FindSymbolContexts(source_location)
        self.assertGreater(target_list.GetSize(), 0)

        module = target.FindModule(lldb.SBFileSpec("a.out"))
        self.assertTrue(module.IsValid())
        module_list = module.FindSymbolContexts(source_location)
        self.assertEqual(target_list.GetSize(), module_list.GetSize())

    def test_invalid_inputs(self):
        """Invalid inputs returns an empty result."""
        self.build()
        target, *_ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        # Default SBLineSpec.
        empty_spec = lldb.SBLineSpec()
        self.assertFalse(empty_spec.IsValid())
        self.assertEqual(target.FindSymbolContexts(empty_spec).GetSize(), 0)

        empty_result = target.FindSymbolContexts(
            lldb.SBLineSpec(lldb.SBFileSpec(), 1)
        )
        self.assertEqual(empty_result.GetSize(), 0)

        missing_line = lldb.SBLineSpec(lldb.SBFileSpec("main.cpp"))
        missing_result = target.FindSymbolContexts(missing_line)
        self.assertEqual(missing_result.GetSize(), 0)
