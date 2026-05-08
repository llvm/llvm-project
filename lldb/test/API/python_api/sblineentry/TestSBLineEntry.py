"""
Test SBLineEntry APIs, particularly synthetic line entries.
"""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SBLineEntryTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_synthetic_line_entry(self):
        """Test that synthetic LineEntry objects (created via SBAPI) can be
        valid without a valid address range and can be set in SBSymbolContext."""

        # Test creating a synthetic line entry via SBAPI.
        line_entry = lldb.SBLineEntry()
        self.assertFalse(
            line_entry.IsValid(), "Default constructed line entry should be invalid"
        )

        # Set line number - this should mark the line entry as synthetic.
        line_entry.SetLine(42)
        self.assertTrue(
            line_entry.IsValid(),
            "Line entry should be valid after setting line, even without address",
        )
        self.assertEqual(line_entry.GetLine(), 42)

        # Set file and column.
        file_spec = lldb.SBFileSpec(os.path.join(self.getSourceDir(), "test.cpp"), True)
        line_entry.SetFileSpec(file_spec)
        line_entry.SetColumn(10)

        self.assertEqual(line_entry.GetColumn(), 10)
        self.assertEqual(line_entry.GetFileSpec().GetFilename(), "test.cpp")

        # Verify address range is still invalid (synthetic).
        start_addr = line_entry.GetStartAddress()
        self.assertFalse(
            start_addr.IsValid(), "Synthetic line entry should not have valid address"
        )

        # Test setting synthetic line entry in symbol context.
        sym_ctx = lldb.SBSymbolContext()
        sym_ctx.SetLineEntry(line_entry)

        retrieved_line_entry = sym_ctx.GetLineEntry()
        self.assertTrue(
            retrieved_line_entry.IsValid(), "Retrieved line entry should be valid"
        )
        self.assertEqual(retrieved_line_entry.GetLine(), 42)
        self.assertEqual(retrieved_line_entry.GetColumn(), 10)
        self.assertEqual(retrieved_line_entry.GetFileSpec().GetFilename(), "test.cpp")

    def test_line_entry_validity_without_address(self):
        """Test that line entries created via SBAPI are valid without addresses."""

        line_entry = lldb.SBLineEntry()

        # Initially invalid.
        self.assertFalse(line_entry.IsValid())

        # Still invalid with just a file spec.
        file_spec = lldb.SBFileSpec("foo.cpp", True)
        line_entry.SetFileSpec(file_spec)
        self.assertFalse(
            line_entry.IsValid(), "Line entry should be invalid without line number"
        )

        # Valid once line number is set (marks as synthetic).
        line_entry.SetLine(100)
        self.assertTrue(
            line_entry.IsValid(), "Line entry should be valid with line number set"
        )

        # Verify no valid address range.
        self.assertFalse(line_entry.GetStartAddress().IsValid())
        self.assertFalse(line_entry.GetEndAddress().IsValid())

    def test_line_entry_column(self):
        """Test setting and getting column information on synthetic line entries."""

        line_entry = lldb.SBLineEntry()
        line_entry.SetLine(50)

        # Default column should be 0.
        self.assertEqual(line_entry.GetColumn(), 0)

        # Set column.
        line_entry.SetColumn(25)
        self.assertEqual(line_entry.GetColumn(), 25)

        # Verify line entry is still valid.
        self.assertTrue(line_entry.IsValid())

    def test_non_synthetic_line_entry_requires_line_number(self):
        """Test that non-synthetic line entries with addresses still require a line number to be valid."""

        # A line entry is always invalid without a line number, regardless of whether it has an address.
        line_entry = lldb.SBLineEntry()
        self.assertFalse(
            line_entry.IsValid(), "Line entry should be invalid without line number"
        )

        # Even with a file spec, it's still invalid.
        file_spec = lldb.SBFileSpec("test.cpp", True)
        line_entry.SetFileSpec(file_spec)
        self.assertFalse(
            line_entry.IsValid(), "Line entry should be invalid without line number"
        )

        # Only after setting a line number does it become valid.
        line_entry.SetLine(42)
        self.assertTrue(
            line_entry.IsValid(), "Line entry should be valid with line number"
        )

    def test_symbol_context_with_synthetic_line_entry(self):
        """Test that SBSymbolContext correctly stores and retrieves synthetic line entries."""

        # Create a synthetic line entry.
        line_entry = lldb.SBLineEntry()
        line_entry.SetLine(123)
        line_entry.SetColumn(45)
        file_spec = lldb.SBFileSpec("source.cpp", True)
        line_entry.SetFileSpec(file_spec)

        # Create symbol context and set line entry.
        sym_ctx = lldb.SBSymbolContext()
        sym_ctx.SetLineEntry(line_entry)

        # Retrieve and verify.
        retrieved = sym_ctx.GetLineEntry()
        self.assertTrue(retrieved.IsValid())
        self.assertEqual(retrieved.GetLine(), 123)
        self.assertEqual(retrieved.GetColumn(), 45)
        self.assertEqual(retrieved.GetFileSpec().GetFilename(), "source.cpp")

        # Verify it's still synthetic (no valid address).
        self.assertFalse(retrieved.GetStartAddress().IsValid())
