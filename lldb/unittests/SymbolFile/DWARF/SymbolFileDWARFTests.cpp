//===-- SymbolFileDWARFTests.cpp --------------------------------*- C++ -*-===//
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
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "Plugins/SymbolFile/PDB/SymbolFilePDB.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataEncoder.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/StreamString.h"



using namespace lldb;
using namespace lldb_private;

class SymbolFileDWARFTests : public testing::Test {
public:
  void SetUp() override {
// Initialize and TearDown the plugin every time, so we get a brand new
// AST every time so that modifications to the AST from each test don't
// leak into the next test.
FileSystem::Initialize();
HostInfo::Initialize();
ObjectFilePECOFF::Initialize();
SymbolFileDWARF::Initialize();
ClangASTContext::Initialize();
SymbolFilePDB::Initialize();

m_dwarf_test_exe = GetInputFilePath("test-dwarf.exe");
  }

  void TearDown() override {
    SymbolFilePDB::Terminate();
    ClangASTContext::Initialize();
    SymbolFileDWARF::Terminate();
    ObjectFilePECOFF::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }

protected:
  std::string m_dwarf_test_exe;
};

TEST_F(SymbolFileDWARFTests, TestAbilitiesForDWARF) {
  // Test that when we have Dwarf debug info, SymbolFileDWARF is used.
  FileSpec fspec(m_dwarf_test_exe);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  EXPECT_NE(nullptr, plugin);
  SymbolFile *symfile = plugin->GetSymbolFile();
  EXPECT_NE(nullptr, symfile);
  EXPECT_EQ(symfile->GetPluginName(), SymbolFileDWARF::GetPluginNameStatic());

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
