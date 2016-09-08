//===-- PythonDataObjectsTests.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/SymbolVendor.h"

#include "Plugins/ObjectFile/PECOFF/ObjectFilePECOFF.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "Plugins/SymbolFile/PDB/SymbolFilePDB.h"

#if defined(_MSC_VER)
#include <objbase.h>
#endif

extern const char *TestMainArgv0;

using namespace lldb_private;

class SymbolFilePDBTests : public testing::Test {
public:
  void SetUp() override {
#if defined(_MSC_VER)
    ::CoInitializeEx(nullptr, COINIT_MULTITHREADED);
#endif

    HostInfoBase::Initialize();
    ObjectFilePECOFF::Initialize();
    SymbolFileDWARF::Initialize();
    SymbolFilePDB::Initialize();

    llvm::StringRef exe_folder = llvm::sys::path::parent_path(TestMainArgv0);
    llvm::SmallString<128> inputs_folder = exe_folder;
    llvm::sys::path::append(inputs_folder, "Inputs");

    m_pdb_test_exe = inputs_folder;
    m_dwarf_test_exe = inputs_folder;
    llvm::sys::path::append(m_pdb_test_exe, "test-pdb.exe");
    llvm::sys::path::append(m_dwarf_test_exe, "test-dwarf.exe");
  }

  void TearDown() override {
#if defined(_MSC_VER)
    ::CoUninitialize();
#endif
    SymbolFilePDB::Terminate();
    SymbolFileDWARF::Terminate();
    ObjectFilePECOFF::Terminate();
  }

protected:
  llvm::SmallString<128> m_pdb_test_exe;
  llvm::SmallString<128> m_dwarf_test_exe;

  bool FileSpecMatchesAsBaseOrFull(const FileSpec &left,
                                   const FileSpec &right) const {
    // If the filenames don't match, the paths can't be equal
    if (!left.FileEquals(right))
      return false;
    // If BOTH have a directory, also compare the directories.
    if (left.GetDirectory() && right.GetDirectory())
      return left.DirectoryEquals(right);

    // If one has a directory but not the other, they match.
    return true;
  }

  void VerifyLineEntry(lldb::ModuleSP module, const SymbolContext &sc,
                       const FileSpec &spec, LineTable &lt, uint32_t line,
                       lldb::addr_t addr) {
    LineEntry entry;
    Address address;
    EXPECT_TRUE(module->ResolveFileAddress(addr, address));

    EXPECT_TRUE(lt.FindLineEntryByAddress(address, entry));
    EXPECT_EQ(line, entry.line);
    EXPECT_EQ(address, entry.range.GetBaseAddress());

    EXPECT_TRUE(FileSpecMatchesAsBaseOrFull(spec, entry.file));
  }

  bool ContainsCompileUnit(const SymbolContextList &sc_list,
                           const FileSpec &spec) const {
    for (size_t i = 0; i < sc_list.GetSize(); ++i) {
      const SymbolContext &sc = sc_list[i];
      if (FileSpecMatchesAsBaseOrFull(*sc.comp_unit, spec))
        return true;
    }
    return false;
  }
};

#if defined(HAVE_DIA_SDK)
#define REQUIRES_DIA_SDK(TestName) TestName
#else
#define REQUIRES_DIA_SDK(TestName) DISABLED_##TestName
#endif

TEST_F(SymbolFilePDBTests, TestAbilitiesForDWARF) {
  // Test that when we have Dwarf debug info, SymbolFileDWARF is used.
  FileSpec fspec(m_dwarf_test_exe.c_str(), false);
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

TEST_F(SymbolFilePDBTests, REQUIRES_DIA_SDK(TestAbilitiesForPDB)) {
  // Test that when we have PDB debug info, SymbolFilePDB is used.
  FileSpec fspec(m_pdb_test_exe.c_str(), false);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  EXPECT_NE(nullptr, plugin);
  SymbolFile *symfile = plugin->GetSymbolFile();
  EXPECT_NE(nullptr, symfile);
  EXPECT_EQ(symfile->GetPluginName(), SymbolFilePDB::GetPluginNameStatic());

  uint32_t expected_abilities =
      SymbolFile::CompileUnits | SymbolFile::LineTables;
  EXPECT_EQ(expected_abilities, symfile->CalculateAbilities());
}

TEST_F(SymbolFilePDBTests, REQUIRES_DIA_SDK(TestResolveSymbolContextBasename)) {
  // Test that attempting to call ResolveSymbolContext with only a basename
  // finds all full paths
  // with the same basename
  FileSpec fspec(m_pdb_test_exe.c_str(), false);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  EXPECT_NE(nullptr, plugin);
  SymbolFile *symfile = plugin->GetSymbolFile();

  FileSpec header_spec("test-pdb.cpp", false);
  SymbolContextList sc_list;
  uint32_t result_count = symfile->ResolveSymbolContext(
      header_spec, 0, false, lldb::eSymbolContextCompUnit, sc_list);
  EXPECT_EQ(1u, result_count);
  EXPECT_TRUE(ContainsCompileUnit(sc_list, header_spec));
}

TEST_F(SymbolFilePDBTests, REQUIRES_DIA_SDK(TestResolveSymbolContextFullPath)) {
  // Test that attempting to call ResolveSymbolContext with a full path only
  // finds the one source
  // file that matches the full path.
  FileSpec fspec(m_pdb_test_exe.c_str(), false);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  EXPECT_NE(nullptr, plugin);
  SymbolFile *symfile = plugin->GetSymbolFile();

  FileSpec header_spec(
      R"spec(D:\src\llvm\tools\lldb\unittests\SymbolFile\PDB\Inputs\test-pdb.cpp)spec",
      false);
  SymbolContextList sc_list;
  uint32_t result_count = symfile->ResolveSymbolContext(
      header_spec, 0, false, lldb::eSymbolContextCompUnit, sc_list);
  EXPECT_GE(1u, result_count);
  EXPECT_TRUE(ContainsCompileUnit(sc_list, header_spec));
}

TEST_F(SymbolFilePDBTests,
       REQUIRES_DIA_SDK(TestLookupOfHeaderFileWithInlines)) {
  // Test that when looking up a header file via ResolveSymbolContext (i.e. a
  // file that was not by itself
  // compiled, but only contributes to the combined code of other source files),
  // a SymbolContext is returned
  // for each compiland which has line contributions from the requested header.
  FileSpec fspec(m_pdb_test_exe.c_str(), false);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  EXPECT_NE(nullptr, plugin);
  SymbolFile *symfile = plugin->GetSymbolFile();

  FileSpec header_specs[] = {FileSpec("test-pdb.h", false),
                             FileSpec("test-pdb-nested.h", false)};
  FileSpec main_cpp_spec("test-pdb.cpp", false);
  FileSpec alt_cpp_spec("test-pdb-alt.cpp", false);
  for (const auto &hspec : header_specs) {
    SymbolContextList sc_list;
    uint32_t result_count = symfile->ResolveSymbolContext(
        hspec, 0, true, lldb::eSymbolContextCompUnit, sc_list);
    EXPECT_EQ(2u, result_count);
    EXPECT_TRUE(ContainsCompileUnit(sc_list, main_cpp_spec));
    EXPECT_TRUE(ContainsCompileUnit(sc_list, alt_cpp_spec));
  }
}

TEST_F(SymbolFilePDBTests,
       REQUIRES_DIA_SDK(TestLookupOfHeaderFileWithNoInlines)) {
  // Test that when looking up a header file via ResolveSymbolContext (i.e. a
  // file that was not by itself
  // compiled, but only contributes to the combined code of other source files),
  // that if check_inlines
  // is false, no SymbolContexts are returned.
  FileSpec fspec(m_pdb_test_exe.c_str(), false);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  EXPECT_NE(nullptr, plugin);
  SymbolFile *symfile = plugin->GetSymbolFile();

  FileSpec header_specs[] = {FileSpec("test-pdb.h", false),
                             FileSpec("test-pdb-nested.h", false)};
  for (const auto &hspec : header_specs) {
    SymbolContextList sc_list;
    uint32_t result_count = symfile->ResolveSymbolContext(
        hspec, 0, false, lldb::eSymbolContextCompUnit, sc_list);
    EXPECT_EQ(0u, result_count);
  }
}

TEST_F(SymbolFilePDBTests, REQUIRES_DIA_SDK(TestLineTablesMatchAll)) {
  // Test that when calling ResolveSymbolContext with a line number of 0, all
  // line entries from
  // the specified files are returned.
  FileSpec fspec(m_pdb_test_exe.c_str(), false);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  SymbolFile *symfile = plugin->GetSymbolFile();

  FileSpec source_file("test-pdb.cpp", false);
  FileSpec header1("test-pdb.h", false);
  FileSpec header2("test-pdb-nested.h", false);
  uint32_t cus = symfile->GetNumCompileUnits();
  EXPECT_EQ(2u, cus);

  SymbolContextList sc_list;
  uint32_t scope = lldb::eSymbolContextCompUnit | lldb::eSymbolContextLineEntry;

  uint32_t count =
      symfile->ResolveSymbolContext(source_file, 0, true, scope, sc_list);
  EXPECT_EQ(1u, count);
  SymbolContext sc;
  EXPECT_TRUE(sc_list.GetContextAtIndex(0, sc));

  LineTable *lt = sc.comp_unit->GetLineTable();
  EXPECT_NE(nullptr, lt);
  count = lt->GetSize();
  // We expect one extra entry for termination (per function)
  EXPECT_EQ(16u, count);

  VerifyLineEntry(module, sc, source_file, *lt, 7, 0x401040);
  VerifyLineEntry(module, sc, source_file, *lt, 8, 0x401043);
  VerifyLineEntry(module, sc, source_file, *lt, 9, 0x401045);

  VerifyLineEntry(module, sc, source_file, *lt, 13, 0x401050);
  VerifyLineEntry(module, sc, source_file, *lt, 14, 0x401054);
  VerifyLineEntry(module, sc, source_file, *lt, 15, 0x401070);

  VerifyLineEntry(module, sc, header1, *lt, 9, 0x401090);
  VerifyLineEntry(module, sc, header1, *lt, 10, 0x401093);
  VerifyLineEntry(module, sc, header1, *lt, 11, 0x4010a2);

  VerifyLineEntry(module, sc, header2, *lt, 5, 0x401080);
  VerifyLineEntry(module, sc, header2, *lt, 6, 0x401083);
  VerifyLineEntry(module, sc, header2, *lt, 7, 0x401089);
}

TEST_F(SymbolFilePDBTests, REQUIRES_DIA_SDK(TestLineTablesMatchSpecific)) {
  // Test that when calling ResolveSymbolContext with a specific line number,
  // only line entries
  // which match the requested line are returned.
  FileSpec fspec(m_pdb_test_exe.c_str(), false);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  SymbolFile *symfile = plugin->GetSymbolFile();

  FileSpec source_file("test-pdb.cpp", false);
  FileSpec header1("test-pdb.h", false);
  FileSpec header2("test-pdb-nested.h", false);
  uint32_t cus = symfile->GetNumCompileUnits();
  EXPECT_EQ(2u, cus);

  SymbolContextList sc_list;
  uint32_t scope = lldb::eSymbolContextCompUnit | lldb::eSymbolContextLineEntry;

  // First test with line 7, and verify that only line 7 entries are added.
  uint32_t count =
      symfile->ResolveSymbolContext(source_file, 7, true, scope, sc_list);
  EXPECT_EQ(1u, count);
  SymbolContext sc;
  EXPECT_TRUE(sc_list.GetContextAtIndex(0, sc));

  LineTable *lt = sc.comp_unit->GetLineTable();
  EXPECT_NE(nullptr, lt);
  count = lt->GetSize();
  // We expect one extra entry for termination
  EXPECT_EQ(3u, count);

  VerifyLineEntry(module, sc, source_file, *lt, 7, 0x401040);
  VerifyLineEntry(module, sc, header2, *lt, 7, 0x401089);

  sc_list.Clear();
  // Then test with line 9, and verify that only line 9 entries are added.
  count = symfile->ResolveSymbolContext(source_file, 9, true, scope, sc_list);
  EXPECT_EQ(1u, count);
  EXPECT_TRUE(sc_list.GetContextAtIndex(0, sc));

  lt = sc.comp_unit->GetLineTable();
  EXPECT_NE(nullptr, lt);
  count = lt->GetSize();
  // We expect one extra entry for termination
  EXPECT_EQ(3u, count);

  VerifyLineEntry(module, sc, source_file, *lt, 9, 0x401045);
  VerifyLineEntry(module, sc, header1, *lt, 9, 0x401090);
}
