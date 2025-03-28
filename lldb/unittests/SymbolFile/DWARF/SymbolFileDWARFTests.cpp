//===-- SymbolFileDWARFTests.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugArangeSet.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "Plugins/ObjectFile/PECOFF/ObjectFilePECOFF.h"
#include "Plugins/SymbolFile/DWARF/DWARFDataExtractor.h"
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
using namespace lldb_private::dwarf;
using namespace lldb_private::plugin::dwarf;
using llvm::DWARFDebugArangeSet;

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
  EXPECT_EQ(symfile->GetPluginName(), SymbolFileDWARF::GetPluginNameStatic());

  uint32_t expected_abilities = SymbolFile::kAllAbilities;
  EXPECT_EQ(expected_abilities, symfile->CalculateAbilities());
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
  llvm::DWARFDataExtractor data(llvm::ArrayRef<unsigned char>(binary_data),
                                /*isLittleEndian=*/false, /*AddrSize=*/4);

  DWARFDebugArangeSet debug_aranges;
  offset_t off = 0;
  llvm::Error error = debug_aranges.extract(data, &off);
  EXPECT_TRUE(bool(error));
  EXPECT_EQ("non-zero segment selector size in address range table at offset "
            "0x0 is not supported",
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
      0, 0, 0, 0,   // unit_length that will be set correctly after this
      0, 2,         // DWARF version number (uint16_t)
      0, 0, 0, 255, // CU offset
      4,            // address size
      0,            // segment size
      0, 0, 0, 0,   // alignment for the first tuple
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
  DWARFDebugAranges debug_aranges;
  debug_aranges.extract(data);
  // Parser should read all terminators to the end of the length specified.
  ASSERT_EQ(debug_aranges.GetNumRanges(), 2U);
  EXPECT_EQ(debug_aranges.FindAddress(0x0fff), DW_INVALID_OFFSET);
  EXPECT_EQ(debug_aranges.FindAddress(0x1000), 255u);
  EXPECT_EQ(debug_aranges.FindAddress(0x1100 - 1), 255u);
  EXPECT_EQ(debug_aranges.FindAddress(0x1100), DW_INVALID_OFFSET);
  EXPECT_EQ(debug_aranges.FindAddress(0x1fff), DW_INVALID_OFFSET);
  EXPECT_EQ(debug_aranges.FindAddress(0x2000), 255u);
  EXPECT_EQ(debug_aranges.FindAddress(0x2010 - 1), 255u);
  EXPECT_EQ(debug_aranges.FindAddress(0x2010), DW_INVALID_OFFSET);
}

TEST_F(SymbolFileDWARFTests, ParseArangesIgnoreEmpty) {
  // This .debug_aranges set has some address ranges which have zero length
  // and we ensure that these are ignored by our DWARFDebugArangeSet parser
  // and not included in the descriptors that are returned.
  unsigned char binary_data[] = {
      0, 0, 0, 0,   // unit_length that will be set correctly after this
      0, 2,         // DWARF version number (uint16_t)
      0, 0, 0, 255, // CU offset
      4,            // address size
      0,            // segment size
      0, 0, 0, 0,   // alignment for the first tuple
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
  DWARFDebugAranges debug_aranges;
  debug_aranges.extract(data);
  // Parser should read all terminators to the end of the length specified.
  // Previously the DWARFDebugArangeSet would stop at the first terminator
  // entry and leave the offset in the middle of the current
  // DWARFDebugArangeSet data, and that would cause the next extracted
  // DWARFDebugArangeSet to fail.
  ASSERT_EQ(debug_aranges.GetNumRanges(), 2U);
  EXPECT_EQ(debug_aranges.FindAddress(0x0fff), DW_INVALID_OFFSET);
  EXPECT_EQ(debug_aranges.FindAddress(0x1000), 255u);
  EXPECT_EQ(debug_aranges.FindAddress(0x1100 - 1), 255u);
  EXPECT_EQ(debug_aranges.FindAddress(0x1100), DW_INVALID_OFFSET);
  EXPECT_EQ(debug_aranges.FindAddress(0x1fff), DW_INVALID_OFFSET);
  EXPECT_EQ(debug_aranges.FindAddress(0x2000), 255u);
  EXPECT_EQ(debug_aranges.FindAddress(0x2010 - 1), 255u);
  EXPECT_EQ(debug_aranges.FindAddress(0x2010), DW_INVALID_OFFSET);
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
