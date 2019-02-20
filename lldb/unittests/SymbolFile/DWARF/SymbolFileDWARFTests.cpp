//===-- PythonDataObjectsTests.cpp ------------------------------*- C++ -*-===//
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
#include "lldb/Utility/FileSpec.h"

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
