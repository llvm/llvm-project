//===-- SymbolFileDWARFTests.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "Plugins/ObjectFile/PECOFF/ObjectFilePECOFF.h"
#include "Plugins/SymbolFile/DWARF/DWARFAbbreviationDeclaration.h"
#include "Plugins/SymbolFile/DWARF/DWARFDataExtractor.h"
#include "Plugins/SymbolFile/DWARF/DWARFDebugAbbrev.h"
#include "Plugins/SymbolFile/DWARF/DWARFDebugArangeSet.h"
#include "Plugins/SymbolFile/DWARF/DWARFDebugAranges.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "Plugins/SymbolFile/PDB/SymbolFilePDB.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataEncoder.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

class SymbolFileDWARFTests : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFilePECOFF, SymbolFileDWARF,
                TypeSystemClang, SymbolFilePDB>
      subsystems;

public:
  void SetUp() override {
    m_dwarf_test_exe = GetInputFilePath("test-dwarf.exe");
  }

protected:
  std::string m_dwarf_test_exe;
};

TEST_F(SymbolFileDWARFTests, TestAbilitiesForDWARF) {
  // Test that when we have Dwarf debug info, SymbolFileDWARF is used.
  FileSpec fspec(m_dwarf_test_exe);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolFile *symfile = module->GetSymbolFile();
  ASSERT_NE(nullptr, symfile);
  EXPECT_EQ(symfile->GetPluginName(),
            SymbolFileDWARF::GetPluginNameStatic().GetStringRef());

  uint32_t expected_abilities = SymbolFile::kAllAbilities;
  EXPECT_EQ(expected_abilities, symfile->CalculateAbilities());
}

TEST_F(SymbolFileDWARFTests, TestAbbrevOrder1Start1) {
  // Test that if we have a .debug_abbrev that contains ordered abbreviation
  // codes that start at 1, that we get O(1) access.

  const auto byte_order = eByteOrderLittle;
  const uint8_t addr_size = 4;
  StreamString encoder(Stream::eBinary, addr_size, byte_order);
  encoder.PutULEB128(1); // Abbrev code 1
  encoder.PutULEB128(DW_TAG_compile_unit);
  encoder.PutHex8(DW_CHILDREN_yes);
  encoder.PutULEB128(DW_AT_name);
  encoder.PutULEB128(DW_FORM_strp);
  encoder.PutULEB128(0);
  encoder.PutULEB128(0);

  encoder.PutULEB128(2); // Abbrev code 2
  encoder.PutULEB128(DW_TAG_subprogram);
  encoder.PutHex8(DW_CHILDREN_no);
  encoder.PutULEB128(DW_AT_name);
  encoder.PutULEB128(DW_FORM_strp);
  encoder.PutULEB128(0);
  encoder.PutULEB128(0);

  encoder.PutULEB128(0); // Abbrev code 0 (termination)

  DWARFDataExtractor data;
  data.SetData(encoder.GetData(), encoder.GetSize(), byte_order);
  DWARFAbbreviationDeclarationSet abbrev_set;
  lldb::offset_t data_offset = 0;
  llvm::Error error = abbrev_set.extract(data, &data_offset);
  EXPECT_FALSE(bool(error));
  // Make sure we have O(1) access to each abbreviation by making sure the
  // index offset is 1 and not UINT32_MAX
  EXPECT_EQ(abbrev_set.GetIndexOffset(), 1u);

  auto abbrev1 = abbrev_set.GetAbbreviationDeclaration(1);
  EXPECT_EQ(abbrev1->Tag(), DW_TAG_compile_unit);
  EXPECT_TRUE(abbrev1->HasChildren());
  EXPECT_EQ(abbrev1->NumAttributes(), 1u);
  auto abbrev2 = abbrev_set.GetAbbreviationDeclaration(2);
  EXPECT_EQ(abbrev2->Tag(), DW_TAG_subprogram);
  EXPECT_FALSE(abbrev2->HasChildren());
  EXPECT_EQ(abbrev2->NumAttributes(), 1u);
}

TEST_F(SymbolFileDWARFTests, TestAbbrevOrder1Start5) {
  // Test that if we have a .debug_abbrev that contains ordered abbreviation
  // codes that start at 5, that we get O(1) access.

  const auto byte_order = eByteOrderLittle;
  const uint8_t addr_size = 4;
  StreamString encoder(Stream::eBinary, addr_size, byte_order);
  encoder.PutULEB128(5); // Abbrev code 5
  encoder.PutULEB128(DW_TAG_compile_unit);
  encoder.PutHex8(DW_CHILDREN_yes);
  encoder.PutULEB128(DW_AT_name);
  encoder.PutULEB128(DW_FORM_strp);
  encoder.PutULEB128(0);
  encoder.PutULEB128(0);

  encoder.PutULEB128(6); // Abbrev code 6
  encoder.PutULEB128(DW_TAG_subprogram);
  encoder.PutHex8(DW_CHILDREN_no);
  encoder.PutULEB128(DW_AT_name);
  encoder.PutULEB128(DW_FORM_strp);
  encoder.PutULEB128(0);
  encoder.PutULEB128(0);

  encoder.PutULEB128(0); // Abbrev code 0 (termination)

  DWARFDataExtractor data;
  data.SetData(encoder.GetData(), encoder.GetSize(), byte_order);
  DWARFAbbreviationDeclarationSet abbrev_set;
  lldb::offset_t data_offset = 0;
  llvm::Error error = abbrev_set.extract(data, &data_offset);
  EXPECT_FALSE(bool(error));
  // Make sure we have O(1) access to each abbreviation by making sure the
  // index offset is 5 and not UINT32_MAX
  EXPECT_EQ(abbrev_set.GetIndexOffset(), 5u);

  auto abbrev1 = abbrev_set.GetAbbreviationDeclaration(5);
  EXPECT_EQ(abbrev1->Tag(), DW_TAG_compile_unit);
  EXPECT_TRUE(abbrev1->HasChildren());
  EXPECT_EQ(abbrev1->NumAttributes(), 1u);
  auto abbrev2 = abbrev_set.GetAbbreviationDeclaration(6);
  EXPECT_EQ(abbrev2->Tag(), DW_TAG_subprogram);
  EXPECT_FALSE(abbrev2->HasChildren());
  EXPECT_EQ(abbrev2->NumAttributes(), 1u);
}

TEST_F(SymbolFileDWARFTests, TestAbbrevOutOfOrder) {
  // Test that if we have a .debug_abbrev that contains unordered abbreviation
  // codes, that we can access the information correctly.

  const auto byte_order = eByteOrderLittle;
  const uint8_t addr_size = 4;
  StreamString encoder(Stream::eBinary, addr_size, byte_order);
  encoder.PutULEB128(2); // Abbrev code 2
  encoder.PutULEB128(DW_TAG_compile_unit);
  encoder.PutHex8(DW_CHILDREN_yes);
  encoder.PutULEB128(DW_AT_name);
  encoder.PutULEB128(DW_FORM_strp);
  encoder.PutULEB128(0);
  encoder.PutULEB128(0);

  encoder.PutULEB128(1); // Abbrev code 1
  encoder.PutULEB128(DW_TAG_subprogram);
  encoder.PutHex8(DW_CHILDREN_no);
  encoder.PutULEB128(DW_AT_name);
  encoder.PutULEB128(DW_FORM_strp);
  encoder.PutULEB128(0);
  encoder.PutULEB128(0);

  encoder.PutULEB128(0); // Abbrev code 0 (termination)

  DWARFDataExtractor data;
  data.SetData(encoder.GetData(), encoder.GetSize(), byte_order);
  DWARFAbbreviationDeclarationSet abbrev_set;
  lldb::offset_t data_offset = 0;
  llvm::Error error = abbrev_set.extract(data, &data_offset);
  EXPECT_FALSE(bool(error));
  // Make sure we don't have O(1) access to each abbreviation by making sure
  // the index offset is UINT32_MAX
  EXPECT_EQ(abbrev_set.GetIndexOffset(), UINT32_MAX);

  auto abbrev1 = abbrev_set.GetAbbreviationDeclaration(2);
  EXPECT_EQ(abbrev1->Tag(), DW_TAG_compile_unit);
  EXPECT_TRUE(abbrev1->HasChildren());
  EXPECT_EQ(abbrev1->NumAttributes(), 1u);
  auto abbrev2 = abbrev_set.GetAbbreviationDeclaration(1);
  EXPECT_EQ(abbrev2->Tag(), DW_TAG_subprogram);
  EXPECT_FALSE(abbrev2->HasChildren());
  EXPECT_EQ(abbrev2->NumAttributes(), 1u);
}

TEST_F(SymbolFileDWARFTests, TestAbbrevInvalidNULLTag) {
  // Test that we detect when an abbreviation has a NULL tag and that we get
  // an error when decoding.

  const auto byte_order = eByteOrderLittle;
  const uint8_t addr_size = 4;
  StreamString encoder(Stream::eBinary, addr_size, byte_order);
  encoder.PutULEB128(1); // Abbrev code 1
  encoder.PutULEB128(0); // Invalid NULL tag here!
  encoder.PutHex8(DW_CHILDREN_no);
  encoder.PutULEB128(0);
  encoder.PutULEB128(0);

  encoder.PutULEB128(0); // Abbrev code 0 (termination)

  DWARFDataExtractor data;
  data.SetData(encoder.GetData(), encoder.GetSize(), byte_order);
  DWARFAbbreviationDeclarationSet abbrev_set;
  lldb::offset_t data_offset = 0;
  llvm::Error error = abbrev_set.extract(data, &data_offset);
  // Verify we get an error
  EXPECT_TRUE(bool(error));
  EXPECT_EQ("abbrev decl requires non-null tag.",
            llvm::toString(std::move(error)));

}

TEST_F(SymbolFileDWARFTests, TestAbbrevNullAttrValidForm) {
  // Test that we detect when an abbreviation has a NULL attribute and a non
  // NULL form and that we get an error when decoding.

  const auto byte_order = eByteOrderLittle;
  const uint8_t addr_size = 4;
  StreamString encoder(Stream::eBinary, addr_size, byte_order);
  encoder.PutULEB128(1); // Abbrev code 1
  encoder.PutULEB128(DW_TAG_compile_unit);
  encoder.PutHex8(DW_CHILDREN_no);
  encoder.PutULEB128(0); // Invalid NULL DW_AT
  encoder.PutULEB128(DW_FORM_strp); // With a valid form
  encoder.PutULEB128(0);
  encoder.PutULEB128(0);

  encoder.PutULEB128(0); // Abbrev code 0 (termination)

  DWARFDataExtractor data;
  data.SetData(encoder.GetData(), encoder.GetSize(), byte_order);
  DWARFAbbreviationDeclarationSet abbrev_set;
  lldb::offset_t data_offset = 0;
  llvm::Error error = abbrev_set.extract(data, &data_offset);
  // Verify we get an error
  EXPECT_TRUE(bool(error));
  EXPECT_EQ("malformed abbreviation declaration attribute",
            llvm::toString(std::move(error)));
}

TEST_F(SymbolFileDWARFTests, TestAbbrevValidAttrNullForm) {
  // Test that we detect when an abbreviation has a valid attribute and a
  // NULL form and that we get an error when decoding.

  const auto byte_order = eByteOrderLittle;
  const uint8_t addr_size = 4;
  StreamString encoder(Stream::eBinary, addr_size, byte_order);
  encoder.PutULEB128(1); // Abbrev code 1
  encoder.PutULEB128(DW_TAG_compile_unit);
  encoder.PutHex8(DW_CHILDREN_no);
  encoder.PutULEB128(DW_AT_name); // Valid attribute
  encoder.PutULEB128(0); // NULL form
  encoder.PutULEB128(0);
  encoder.PutULEB128(0);

  encoder.PutULEB128(0); // Abbrev code 0 (termination)

  DWARFDataExtractor data;
  data.SetData(encoder.GetData(), encoder.GetSize(), byte_order);
  DWARFAbbreviationDeclarationSet abbrev_set;
  lldb::offset_t data_offset = 0;
  llvm::Error error = abbrev_set.extract(data, &data_offset);
  // Verify we get an error
  EXPECT_TRUE(bool(error));
  EXPECT_EQ("malformed abbreviation declaration attribute",
            llvm::toString(std::move(error)));
}

TEST_F(SymbolFileDWARFTests, TestAbbrevMissingTerminator) {
  // Test that we detect when an abbreviation has a valid attribute and a
  // form, but is missing the NULL attribute and form that terminates an
  // abbreviation

  const auto byte_order = eByteOrderLittle;
  const uint8_t addr_size = 4;
  StreamString encoder(Stream::eBinary, addr_size, byte_order);
  encoder.PutULEB128(1); // Abbrev code 1
  encoder.PutULEB128(DW_TAG_compile_unit);
  encoder.PutHex8(DW_CHILDREN_no);
  encoder.PutULEB128(DW_AT_name);
  encoder.PutULEB128(DW_FORM_strp);
  // Don't add the NULL DW_AT and NULL DW_FORM terminator

  DWARFDataExtractor data;
  data.SetData(encoder.GetData(), encoder.GetSize(), byte_order);
  DWARFAbbreviationDeclarationSet abbrev_set;
  lldb::offset_t data_offset = 0;
  llvm::Error error = abbrev_set.extract(data, &data_offset);
  // Verify we get an error
  EXPECT_TRUE(bool(error));
  EXPECT_EQ("abbreviation declaration attribute list not terminated with a "
            "null entry", llvm::toString(std::move(error)));
}

TEST_F(SymbolFileDWARFTests, ParseArangesNonzeroSegmentSize) {
  // This `.debug_aranges` table header is a valid 32bit big-endian section
  // according to the DWARFv5 spec:6.2.1, but contains segment selectors which
  // are not supported by lldb, and should be gracefully rejected
  const unsigned char binary_data[] = {
      0, 0, 0, 41, // unit_length (length field not including this field itself)
      0, 2,        // DWARF version number (half)
      0, 0, 0, 0, // offset into the .debug_info_table (ignored for the purposes
                  // of this test
      4,          // address size
      1,          // segment size
      // alignment for the first tuple which "begins at an offset that is a
      // multiple of the size of a single tuple". Tuples are nine bytes in this
      // example.
      0, 0, 0, 0, 0, 0,
      // BEGIN TUPLES
      1, 0, 0, 0, 4, 0, 0, 0,
      1, // a 1byte object starting at address 4 in segment 1
      0, 0, 0, 0, 4, 0, 0, 0,
      1, // a 1byte object starting at address 4 in segment 0
      // END TUPLES
      0, 0, 0, 0, 0, 0, 0, 0, 0 // terminator
  };
  DWARFDataExtractor data;
  data.SetData(static_cast<const void *>(binary_data), sizeof binary_data,
               lldb::ByteOrder::eByteOrderBig);
  DWARFDebugArangeSet debug_aranges;
  offset_t off = 0;
  llvm::Error error = debug_aranges.extract(data, &off);
  EXPECT_TRUE(bool(error));
  EXPECT_EQ("segmented arange entries are not supported",
            llvm::toString(std::move(error)));
  EXPECT_EQ(off, 12U); // Parser should read no further than the segment size
}

TEST_F(SymbolFileDWARFTests, ParseArangesWithMultipleTerminators) {
  // This .debug_aranges set has multiple terminator entries which appear in
  // binaries produced by popular linux compilers and linker combinations. We
  // must be able to parse all the way through the data for each
  // DWARFDebugArangeSet. Previously the DWARFDebugArangeSet::extract()
  // function would stop parsing as soon as we ran into a terminator even
  // though the length field stated that there was more data that follows. This
  // would cause the next DWARFDebugArangeSet to be parsed immediately
  // following the first terminator and it would attempt to decode the
  // DWARFDebugArangeSet header using the remaining segment + address pairs
  // from the remaining bytes.
  unsigned char binary_data[] = {
      0, 0, 0, 0, // unit_length that will be set correctly after this
      0, 2,       // DWARF version number (uint16_t)
      0, 0, 0, 0, // CU offset (ignored for the purposes of this test)
      4,          // address size
      0,          // segment size
      0, 0, 0, 0, // alignment for the first tuple
      // BEGIN TUPLES
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // premature terminator
      0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, // [0x1000-0x1100)
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // premature terminator
      0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x10, // [0x2000-0x2010)
      // END TUPLES
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // terminator
  };
  // Set the big endian length correctly.
  const offset_t binary_data_size = sizeof(binary_data);
  binary_data[3] = (uint8_t)binary_data_size - 4;
  DWARFDataExtractor data;
  data.SetData(static_cast<const void *>(binary_data), sizeof binary_data,
               lldb::ByteOrder::eByteOrderBig);
  DWARFDebugArangeSet set;
  offset_t off = 0;
  llvm::Error error = set.extract(data, &off);
  // Multiple terminators are not fatal as they do appear in binaries.
  EXPECT_FALSE(bool(error));
  // Parser should read all terminators to the end of the length specified.
  EXPECT_EQ(off, binary_data_size);
  ASSERT_EQ(set.NumDescriptors(), 2U);
  ASSERT_EQ(set.GetDescriptorRef(0).address, (dw_addr_t)0x1000);
  ASSERT_EQ(set.GetDescriptorRef(0).length, (dw_addr_t)0x100);
  ASSERT_EQ(set.GetDescriptorRef(1).address, (dw_addr_t)0x2000);
  ASSERT_EQ(set.GetDescriptorRef(1).length, (dw_addr_t)0x10);
}

TEST_F(SymbolFileDWARFTests, ParseArangesIgnoreEmpty) {
  // This .debug_aranges set has some address ranges which have zero length
  // and we ensure that these are ignored by our DWARFDebugArangeSet parser
  // and not included in the descriptors that are returned.
  unsigned char binary_data[] = {
      0, 0, 0, 0, // unit_length that will be set correctly after this
      0, 2,       // DWARF version number (uint16_t)
      0, 0, 0, 0, // CU offset (ignored for the purposes of this test)
      4,          // address size
      0,          // segment size
      0, 0, 0, 0, // alignment for the first tuple
      // BEGIN TUPLES
      0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, // [0x1000-0x1100)
      0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, // [0x1100-0x1100)
      0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x10, // [0x2000-0x2010)
      0x00, 0x00, 0x20, 0x10, 0x00, 0x00, 0x00, 0x00, // [0x2010-0x2010)
      // END TUPLES
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // terminator
  };
  // Set the big endian length correctly.
  const offset_t binary_data_size = sizeof(binary_data);
  binary_data[3] = (uint8_t)binary_data_size - 4;
  DWARFDataExtractor data;
  data.SetData(static_cast<const void *>(binary_data), sizeof binary_data,
               lldb::ByteOrder::eByteOrderBig);
  DWARFDebugArangeSet set;
  offset_t off = 0;
  llvm::Error error = set.extract(data, &off);
  // Multiple terminators are not fatal as they do appear in binaries.
  EXPECT_FALSE(bool(error));
  // Parser should read all terminators to the end of the length specified.
  // Previously the DWARFDebugArangeSet would stop at the first terminator
  // entry and leave the offset in the middle of the current
  // DWARFDebugArangeSet data, and that would cause the next extracted
  // DWARFDebugArangeSet to fail.
  EXPECT_EQ(off, binary_data_size);
  ASSERT_EQ(set.NumDescriptors(), 2U);
  ASSERT_EQ(set.GetDescriptorRef(0).address, (dw_addr_t)0x1000);
  ASSERT_EQ(set.GetDescriptorRef(0).length, (dw_addr_t)0x100);
  ASSERT_EQ(set.GetDescriptorRef(1).address, (dw_addr_t)0x2000);
  ASSERT_EQ(set.GetDescriptorRef(1).length, (dw_addr_t)0x10);
}

TEST_F(SymbolFileDWARFTests, ParseAranges) {
  // Test we can successfully parse a DWARFDebugAranges. The initial error
  // checking code had a bug where it would always return an empty address
  // ranges for everything in .debug_aranges and no error.
  unsigned char binary_data[] = {
      0, 0, 0, 0,   // unit_length that will be set correctly after this
      2, 0,         // DWARF version number
      255, 0, 0, 0, // offset into the .debug_info_table
      8,            // address size
      0,            // segment size
      0, 0, 0, 0,   // pad bytes
      // BEGIN TUPLES
      // First tuple: [0x1000-0x1100)
      0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Address 0x1000
      0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Size    0x0100
      // Second tuple: [0x2000-0x2100)
      0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Address 0x2000
      0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Size    0x0100
      // Terminating tuple
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Terminator
  };
  // Set the little endian length correctly.
  binary_data[0] = sizeof(binary_data) - 4;
  DWARFDataExtractor data;
  data.SetData(static_cast<const void *>(binary_data), sizeof binary_data,
               lldb::ByteOrder::eByteOrderLittle);
  DWARFDebugAranges debug_aranges;
  debug_aranges.extract(data);
  EXPECT_EQ(debug_aranges.GetNumRanges(), 2u);
  EXPECT_EQ(debug_aranges.FindAddress(0x0fff), DW_INVALID_OFFSET);
  EXPECT_EQ(debug_aranges.FindAddress(0x1000), 255u);
  EXPECT_EQ(debug_aranges.FindAddress(0x1100 - 1), 255u);
  EXPECT_EQ(debug_aranges.FindAddress(0x1100), DW_INVALID_OFFSET);
  EXPECT_EQ(debug_aranges.FindAddress(0x1fff), DW_INVALID_OFFSET);
  EXPECT_EQ(debug_aranges.FindAddress(0x2000), 255u);
  EXPECT_EQ(debug_aranges.FindAddress(0x2100 - 1), 255u);
  EXPECT_EQ(debug_aranges.FindAddress(0x2100), DW_INVALID_OFFSET);
}

TEST_F(SymbolFileDWARFTests, ParseArangesSkipErrors) {
  // Test we can successfully parse a DWARFDebugAranges that contains some
  // valid DWARFDebugArangeSet objects and some with errors as long as their
  // length is set correctly. This helps LLDB ensure that it can parse newer
  // .debug_aranges version that LLDB currently doesn't support, or ignore
  // errors in individual DWARFDebugArangeSet objects as long as the length
  // is set correctly.
  const unsigned char binary_data[] = {
      // This DWARFDebugArangeSet is well formed and has a single address range
      // for [0x1000-0x1100) with a CU offset of 0x00000000.
      0, 0, 0, 28, // unit_length that will be set correctly after this
      0, 2,        // DWARF version number (uint16_t)
      0, 0, 0, 0,  // CU offset = 0x00000000
      4,           // address size
      0,           // segment size
      0, 0, 0, 0,  // alignment for the first tuple
      0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, // [0x1000-0x1100)
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // terminator
      // This DWARFDebugArangeSet has the correct length, but an invalid
      // version. We need to be able to skip this correctly and ignore it.
      0, 0, 0, 20, // unit_length that will be set correctly after this
      0, 44,       // invalid DWARF version number (uint16_t)
      0, 0, 1, 0,  // CU offset = 0x00000100
      4,           // address size
      0,           // segment size
      0, 0, 0, 0,  // alignment for the first tuple
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // terminator
      // This DWARFDebugArangeSet is well formed and has a single address range
      // for [0x2000-0x2100) with a CU offset of 0x00000000.
      0, 0, 0, 28, // unit_length that will be set correctly after this
      0, 2,        // DWARF version number (uint16_t)
      0, 0, 2, 0,  // CU offset = 0x00000200
      4,           // address size
      0,           // segment size
      0, 0, 0, 0,  // alignment for the first tuple
      0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x01, 0x00, // [0x2000-0x2100)
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // terminator
  };

  DWARFDataExtractor data;
  data.SetData(static_cast<const void *>(binary_data), sizeof binary_data,
               lldb::ByteOrder::eByteOrderBig);
  DWARFDebugAranges debug_aranges;
  debug_aranges.extract(data);
  EXPECT_EQ(debug_aranges.GetNumRanges(), 2u);
  EXPECT_EQ(debug_aranges.FindAddress(0x0fff), DW_INVALID_OFFSET);
  EXPECT_EQ(debug_aranges.FindAddress(0x1000), 0u);
  EXPECT_EQ(debug_aranges.FindAddress(0x1100 - 1), 0u);
  EXPECT_EQ(debug_aranges.FindAddress(0x1100), DW_INVALID_OFFSET);
  EXPECT_EQ(debug_aranges.FindAddress(0x1fff), DW_INVALID_OFFSET);
  EXPECT_EQ(debug_aranges.FindAddress(0x2000), 0x200u);
  EXPECT_EQ(debug_aranges.FindAddress(0x2100 - 1), 0x200u);
  EXPECT_EQ(debug_aranges.FindAddress(0x2100), DW_INVALID_OFFSET);
}
