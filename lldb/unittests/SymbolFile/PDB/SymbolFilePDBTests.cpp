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

#if defined(_MSC_VER)
#include "lldb/Host/windows/windows.h"
#include <objbase.h>
#endif

#include <algorithm>

using namespace lldb_private;

class SymbolFilePDBTests : public testing::Test {
public:
  void SetUp() override {
// Initialize and TearDown the plugin every time, so we get a brand new
// AST every time so that modifications to the AST from each test don't
// leak into the next test.
#if defined(_MSC_VER)
    ::CoInitializeEx(nullptr, COINIT_MULTITHREADED);
#endif

    FileSystem::Initialize();
    HostInfo::Initialize();
    ObjectFilePECOFF::Initialize();
    SymbolFileDWARF::Initialize();
    ClangASTContext::Initialize();
    SymbolFilePDB::Initialize();

    m_pdb_test_exe = GetInputFilePath("test-pdb.exe");
    m_types_test_exe = GetInputFilePath("test-pdb-types.exe");
  }

  void TearDown() override {
    SymbolFilePDB::Terminate();
    ClangASTContext::Initialize();
    SymbolFileDWARF::Terminate();
    ObjectFilePECOFF::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();

#if defined(_MSC_VER)
    ::CoUninitialize();
#endif
  }

protected:
  std::string m_pdb_test_exe;
  std::string m_types_test_exe;

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

  uint64_t GetGlobalConstantInteger(llvm::pdb::IPDBSession &session,
                                    llvm::StringRef var) const {
    auto global = session.getGlobalScope();
    auto results =
        global->findChildren(llvm::pdb::PDB_SymType::Data, var,
                             llvm::pdb::PDB_NameSearchFlags::NS_Default);
    uint32_t count = results->getChildCount();
    if (count == 0)
      return -1;

    auto item = results->getChildAtIndex(0);
    auto symbol = llvm::dyn_cast<llvm::pdb::PDBSymbolData>(item.get());
    if (!symbol)
      return -1;
    llvm::pdb::Variant value = symbol->getValue();
    switch (value.Type) {
    case llvm::pdb::PDB_VariantType::Int16:
      return value.Value.Int16;
    case llvm::pdb::PDB_VariantType::Int32:
      return value.Value.Int32;
    case llvm::pdb::PDB_VariantType::UInt16:
      return value.Value.UInt16;
    case llvm::pdb::PDB_VariantType::UInt32:
      return value.Value.UInt32;
    default:
      return 0;
    }
  }
};

TEST_F(SymbolFilePDBTests, TestAbilitiesForPDB) {
  // Test that when we have PDB debug info, SymbolFilePDB is used.
  FileSpec fspec(m_pdb_test_exe);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  EXPECT_NE(nullptr, plugin);
  SymbolFile *symfile = plugin->GetSymbolFile();
  EXPECT_NE(nullptr, symfile);
  EXPECT_EQ(symfile->GetPluginName(), SymbolFilePDB::GetPluginNameStatic());

  uint32_t expected_abilities = SymbolFile::kAllAbilities;
  EXPECT_EQ(expected_abilities, symfile->CalculateAbilities());
}

TEST_F(SymbolFilePDBTests, TestResolveSymbolContextBasename) {
  // Test that attempting to call ResolveSymbolContext with only a basename
  // finds all full paths
  // with the same basename
  FileSpec fspec(m_pdb_test_exe);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  EXPECT_NE(nullptr, plugin);
  SymbolFile *symfile = plugin->GetSymbolFile();

  FileSpec header_spec("test-pdb.cpp");
  SymbolContextList sc_list;
  uint32_t result_count = symfile->ResolveSymbolContext(
      header_spec, 0, false, lldb::eSymbolContextCompUnit, sc_list);
  EXPECT_EQ(1u, result_count);
  EXPECT_TRUE(ContainsCompileUnit(sc_list, header_spec));
}

TEST_F(SymbolFilePDBTests, TestResolveSymbolContextFullPath) {
  // Test that attempting to call ResolveSymbolContext with a full path only
  // finds the one source
  // file that matches the full path.
  FileSpec fspec(m_pdb_test_exe);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  EXPECT_NE(nullptr, plugin);
  SymbolFile *symfile = plugin->GetSymbolFile();

  FileSpec header_spec(
      R"spec(D:\src\llvm\tools\lldb\unittests\SymbolFile\PDB\Inputs\test-pdb.cpp)spec");
  SymbolContextList sc_list;
  uint32_t result_count = symfile->ResolveSymbolContext(
      header_spec, 0, false, lldb::eSymbolContextCompUnit, sc_list);
  EXPECT_GE(1u, result_count);
  EXPECT_TRUE(ContainsCompileUnit(sc_list, header_spec));
}

TEST_F(SymbolFilePDBTests, TestLookupOfHeaderFileWithInlines) {
  // Test that when looking up a header file via ResolveSymbolContext (i.e. a
  // file that was not by itself
  // compiled, but only contributes to the combined code of other source files),
  // a SymbolContext is returned
  // for each compiland which has line contributions from the requested header.
  FileSpec fspec(m_pdb_test_exe);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  EXPECT_NE(nullptr, plugin);
  SymbolFile *symfile = plugin->GetSymbolFile();

  FileSpec header_specs[] = {FileSpec("test-pdb.h"),
                             FileSpec("test-pdb-nested.h")};
  FileSpec main_cpp_spec("test-pdb.cpp");
  FileSpec alt_cpp_spec("test-pdb-alt.cpp");
  for (const auto &hspec : header_specs) {
    SymbolContextList sc_list;
    uint32_t result_count = symfile->ResolveSymbolContext(
        hspec, 0, true, lldb::eSymbolContextCompUnit, sc_list);
    EXPECT_EQ(2u, result_count);
    EXPECT_TRUE(ContainsCompileUnit(sc_list, main_cpp_spec));
    EXPECT_TRUE(ContainsCompileUnit(sc_list, alt_cpp_spec));
  }
}

TEST_F(SymbolFilePDBTests, TestLookupOfHeaderFileWithNoInlines) {
  // Test that when looking up a header file via ResolveSymbolContext (i.e. a
  // file that was not by itself
  // compiled, but only contributes to the combined code of other source files),
  // that if check_inlines
  // is false, no SymbolContexts are returned.
  FileSpec fspec(m_pdb_test_exe);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  EXPECT_NE(nullptr, plugin);
  SymbolFile *symfile = plugin->GetSymbolFile();

  FileSpec header_specs[] = {FileSpec("test-pdb.h"),
                             FileSpec("test-pdb-nested.h")};
  for (const auto &hspec : header_specs) {
    SymbolContextList sc_list;
    uint32_t result_count = symfile->ResolveSymbolContext(
        hspec, 0, false, lldb::eSymbolContextCompUnit, sc_list);
    EXPECT_EQ(0u, result_count);
  }
}

TEST_F(SymbolFilePDBTests, TestLineTablesMatchAll) {
  // Test that when calling ResolveSymbolContext with a line number of 0, all
  // line entries from
  // the specified files are returned.
  FileSpec fspec(m_pdb_test_exe);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  SymbolFile *symfile = plugin->GetSymbolFile();

  FileSpec source_file("test-pdb.cpp");
  FileSpec header1("test-pdb.h");
  FileSpec header2("test-pdb-nested.h");
  uint32_t cus = symfile->GetNumCompileUnits();
  EXPECT_EQ(2u, cus);

  SymbolContextList sc_list;
  lldb::SymbolContextItem scope =
      lldb::eSymbolContextCompUnit | lldb::eSymbolContextLineEntry;

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

TEST_F(SymbolFilePDBTests, TestLineTablesMatchSpecific) {
  // Test that when calling ResolveSymbolContext with a specific line number,
  // only line entries
  // which match the requested line are returned.
  FileSpec fspec(m_pdb_test_exe);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  SymbolFile *symfile = plugin->GetSymbolFile();

  FileSpec source_file("test-pdb.cpp");
  FileSpec header1("test-pdb.h");
  FileSpec header2("test-pdb-nested.h");
  uint32_t cus = symfile->GetNumCompileUnits();
  EXPECT_EQ(2u, cus);

  SymbolContextList sc_list;
  lldb::SymbolContextItem scope =
      lldb::eSymbolContextCompUnit | lldb::eSymbolContextLineEntry;

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

TEST_F(SymbolFilePDBTests, TestSimpleClassTypes) {
  FileSpec fspec(m_types_test_exe);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  SymbolFilePDB *symfile =
      static_cast<SymbolFilePDB *>(plugin->GetSymbolFile());
  llvm::pdb::IPDBSession &session = symfile->GetPDBSession();
  llvm::DenseSet<SymbolFile *> searched_files;
  TypeMap results;
  EXPECT_EQ(1u, symfile->FindTypes(ConstString("Class"), nullptr, false, 0,
                                   searched_files, results));
  EXPECT_EQ(1u, results.GetSize());
  lldb::TypeSP udt_type = results.GetTypeAtIndex(0);
  EXPECT_EQ(ConstString("Class"), udt_type->GetName());
  CompilerType compiler_type = udt_type->GetForwardCompilerType();
  EXPECT_TRUE(ClangASTContext::IsClassType(compiler_type.GetOpaqueQualType()));
  EXPECT_EQ(GetGlobalConstantInteger(session, "sizeof_Class"),
            udt_type->GetByteSize());
}

TEST_F(SymbolFilePDBTests, TestNestedClassTypes) {
  FileSpec fspec(m_types_test_exe);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  SymbolFilePDB *symfile =
      static_cast<SymbolFilePDB *>(plugin->GetSymbolFile());
  llvm::pdb::IPDBSession &session = symfile->GetPDBSession();
  llvm::DenseSet<SymbolFile *> searched_files;
  TypeMap results;

  auto clang_ast_ctx = llvm::dyn_cast_or_null<ClangASTContext>(
      symfile->GetTypeSystemForLanguage(lldb::eLanguageTypeC_plus_plus));
  EXPECT_NE(nullptr, clang_ast_ctx);

  EXPECT_EQ(1u, symfile->FindTypes(ConstString("Class"), nullptr, false, 0,
                                   searched_files, results));
  EXPECT_EQ(1u, results.GetSize());

  auto Class = results.GetTypeAtIndex(0);
  EXPECT_TRUE(Class);
  EXPECT_TRUE(Class->IsValidType());

  auto ClassCompilerType = Class->GetFullCompilerType();
  EXPECT_TRUE(ClassCompilerType.IsValid());

  auto ClassDeclCtx = clang_ast_ctx->GetDeclContextForType(ClassCompilerType);
  EXPECT_NE(nullptr, ClassDeclCtx);

  // There are two symbols for nested classes: one belonging to enclosing class
  // and one is global. We process correctly this case and create the same
  // compiler type for both, but `FindTypes` may return more than one type
  // (with the same compiler type) because the symbols have different IDs.
  auto ClassCompilerDeclCtx = CompilerDeclContext(clang_ast_ctx, ClassDeclCtx);
  EXPECT_LE(1u, symfile->FindTypes(ConstString("NestedClass"),
                                   &ClassCompilerDeclCtx, false, 0,
                                   searched_files, results));
  EXPECT_LE(1u, results.GetSize());

  lldb::TypeSP udt_type = results.GetTypeAtIndex(0);
  EXPECT_EQ(ConstString("NestedClass"), udt_type->GetName());

  CompilerType compiler_type = udt_type->GetForwardCompilerType();
  EXPECT_TRUE(ClangASTContext::IsClassType(compiler_type.GetOpaqueQualType()));

  EXPECT_EQ(GetGlobalConstantInteger(session, "sizeof_NestedClass"),
            udt_type->GetByteSize());
}

TEST_F(SymbolFilePDBTests, TestClassInNamespace) {
  FileSpec fspec(m_types_test_exe);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  SymbolFilePDB *symfile =
      static_cast<SymbolFilePDB *>(plugin->GetSymbolFile());
  llvm::pdb::IPDBSession &session = symfile->GetPDBSession();
  llvm::DenseSet<SymbolFile *> searched_files;
  TypeMap results;

  auto clang_ast_ctx = llvm::dyn_cast_or_null<ClangASTContext>(
      symfile->GetTypeSystemForLanguage(lldb::eLanguageTypeC_plus_plus));
  EXPECT_NE(nullptr, clang_ast_ctx);

  auto ast_ctx = clang_ast_ctx->getASTContext();
  EXPECT_NE(nullptr, ast_ctx);

  auto tu = ast_ctx->getTranslationUnitDecl();
  EXPECT_NE(nullptr, tu);

  symfile->ParseDeclsForContext(CompilerDeclContext(
      clang_ast_ctx, static_cast<clang::DeclContext *>(tu)));

  auto ns_namespace = symfile->FindNamespace(ConstString("NS"), nullptr);
  EXPECT_TRUE(ns_namespace.IsValid());

  EXPECT_EQ(1u, symfile->FindTypes(ConstString("NSClass"), &ns_namespace, false,
                                   0, searched_files, results));
  EXPECT_EQ(1u, results.GetSize());

  lldb::TypeSP udt_type = results.GetTypeAtIndex(0);
  EXPECT_EQ(ConstString("NSClass"), udt_type->GetName());

  CompilerType compiler_type = udt_type->GetForwardCompilerType();
  EXPECT_TRUE(ClangASTContext::IsClassType(compiler_type.GetOpaqueQualType()));

  EXPECT_EQ(GetGlobalConstantInteger(session, "sizeof_NSClass"),
            udt_type->GetByteSize());
}

TEST_F(SymbolFilePDBTests, TestEnumTypes) {
  FileSpec fspec(m_types_test_exe);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  SymbolFilePDB *symfile =
      static_cast<SymbolFilePDB *>(plugin->GetSymbolFile());
  llvm::pdb::IPDBSession &session = symfile->GetPDBSession();
  llvm::DenseSet<SymbolFile *> searched_files;
  const char *EnumsToCheck[] = {"Enum", "ShortEnum"};
  for (auto Enum : EnumsToCheck) {
    TypeMap results;
    EXPECT_EQ(1u, symfile->FindTypes(ConstString(Enum), nullptr, false, 0,
                                     searched_files, results));
    EXPECT_EQ(1u, results.GetSize());
    lldb::TypeSP enum_type = results.GetTypeAtIndex(0);
    EXPECT_EQ(ConstString(Enum), enum_type->GetName());
    CompilerType compiler_type = enum_type->GetFullCompilerType();
    EXPECT_TRUE(ClangASTContext::IsEnumType(compiler_type.GetOpaqueQualType()));
    clang::EnumDecl *enum_decl = ClangASTContext::GetAsEnumDecl(compiler_type);
    EXPECT_NE(nullptr, enum_decl);
    EXPECT_EQ(2, std::distance(enum_decl->enumerator_begin(),
                               enum_decl->enumerator_end()));

    std::string sizeof_var = "sizeof_";
    sizeof_var.append(Enum);
    EXPECT_EQ(GetGlobalConstantInteger(session, sizeof_var),
              enum_type->GetByteSize());
  }
}

TEST_F(SymbolFilePDBTests, TestArrayTypes) {
  // In order to get this test working, we need to support lookup by symbol
  // name.  Because array
  // types themselves do not have names, only the symbols have names (i.e. the
  // name of the array).
}

TEST_F(SymbolFilePDBTests, TestFunctionTypes) {
  // In order to get this test working, we need to support lookup by symbol
  // name.  Because array
  // types themselves do not have names, only the symbols have names (i.e. the
  // name of the array).
}

TEST_F(SymbolFilePDBTests, TestTypedefs) {
  FileSpec fspec(m_types_test_exe);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  SymbolFilePDB *symfile =
      static_cast<SymbolFilePDB *>(plugin->GetSymbolFile());
  llvm::pdb::IPDBSession &session = symfile->GetPDBSession();
  llvm::DenseSet<SymbolFile *> searched_files;
  TypeMap results;

  const char *TypedefsToCheck[] = {"ClassTypedef", "NSClassTypedef",
                                   "FuncPointerTypedef",
                                   "VariadicFuncPointerTypedef"};
  for (auto Typedef : TypedefsToCheck) {
    TypeMap results;
    EXPECT_EQ(1u, symfile->FindTypes(ConstString(Typedef), nullptr, false, 0,
                                     searched_files, results));
    EXPECT_EQ(1u, results.GetSize());
    lldb::TypeSP typedef_type = results.GetTypeAtIndex(0);
    EXPECT_EQ(ConstString(Typedef), typedef_type->GetName());
    CompilerType compiler_type = typedef_type->GetFullCompilerType();
    ClangASTContext *clang_type_system =
        llvm::dyn_cast_or_null<ClangASTContext>(compiler_type.GetTypeSystem());
    EXPECT_TRUE(
        clang_type_system->IsTypedefType(compiler_type.GetOpaqueQualType()));

    std::string sizeof_var = "sizeof_";
    sizeof_var.append(Typedef);
    EXPECT_EQ(GetGlobalConstantInteger(session, sizeof_var),
              typedef_type->GetByteSize());
  }
}

TEST_F(SymbolFilePDBTests, TestRegexNameMatch) {
  FileSpec fspec(m_types_test_exe);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  SymbolFilePDB *symfile =
      static_cast<SymbolFilePDB *>(plugin->GetSymbolFile());
  TypeMap results;

  symfile->FindTypesByRegex(RegularExpression(".*"), 0, results);
  EXPECT_GT(results.GetSize(), 1u);

  // We expect no exception thrown if the given regex can't be compiled
  results.Clear();
  symfile->FindTypesByRegex(RegularExpression("**"), 0, results);
  EXPECT_EQ(0u, results.GetSize());
}

TEST_F(SymbolFilePDBTests, TestMaxMatches) {
  FileSpec fspec(m_types_test_exe);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  SymbolFilePDB *symfile =
      static_cast<SymbolFilePDB *>(plugin->GetSymbolFile());
  llvm::DenseSet<SymbolFile *> searched_files;
  TypeMap results;
  const ConstString name("ClassTypedef");
  uint32_t num_results =
      symfile->FindTypes(name, nullptr, false, 0, searched_files, results);
  // Try to limit ourselves from 1 to 10 results, otherwise we could be doing
  // this thousands of times.
  // The idea is just to make sure that for a variety of values, the number of
  // limited results always
  // comes out to the number we are expecting.
  uint32_t iterations = std::min(num_results, 10u);
  for (uint32_t i = 1; i <= iterations; ++i) {
    uint32_t num_limited_results =
        symfile->FindTypes(name, nullptr, false, i, searched_files, results);
    EXPECT_EQ(i, num_limited_results);
    EXPECT_EQ(num_limited_results, results.GetSize());
  }
}

TEST_F(SymbolFilePDBTests, TestNullName) {
  FileSpec fspec(m_types_test_exe);
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolVendor *plugin = module->GetSymbolVendor();
  SymbolFilePDB *symfile =
      static_cast<SymbolFilePDB *>(plugin->GetSymbolFile());
  llvm::DenseSet<SymbolFile *> searched_files;
  TypeMap results;
  uint32_t num_results = symfile->FindTypes(ConstString(), nullptr, false, 0,
                                            searched_files, results);
  EXPECT_EQ(0u, num_results);
  EXPECT_EQ(0u, results.GetSize());
}

TEST_F(SymbolFilePDBTests, TestFindSymbolsWithNameAndType) {
  FileSpec fspec(m_pdb_test_exe.c_str());
  ArchSpec aspec("i686-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolContextList sc_list;
  EXPECT_EQ(1u,
            module->FindSymbolsWithNameAndType(ConstString("?foo@@YAHH@Z"),
                                               lldb::eSymbolTypeAny, sc_list));
  EXPECT_EQ(1u, sc_list.GetSize());

  SymbolContext sc;
  EXPECT_TRUE(sc_list.GetContextAtIndex(0, sc));
  EXPECT_STREQ("int foo(int)",
               sc.GetFunctionName(Mangled::ePreferDemangled).AsCString());
}
