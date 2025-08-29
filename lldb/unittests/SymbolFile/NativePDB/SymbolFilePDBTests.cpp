//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ExpressionParser/Clang/ClangASTMetadata.h"
#include "Plugins/ObjectFile/PECOFF/ObjectFilePECOFF.h"
#include "Plugins/Platform/Windows/PlatformWindows.h"
#include "Plugins/SymbolFile/NativePDB/SymbolFileNativePDB.h"
#include "Plugins/SymbolFile/NativePDB/UdtRecordCompleter.h"
#include "Plugins/SymbolFile/PDB/SymbolFilePDB.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"

#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Platform.h"
#include "lldb/Utility/ArchSpec.h"

#include "gtest/gtest.h"

#include <optional>


using namespace lldb_private;
using namespace lldb_private::npdb;
using namespace llvm;

class SymbolFilePDBTests : public testing::Test {
public:
  void SetUp() override {
    m_test_exe = GetInputFilePath("DynamicTypes.exe");

    ArchSpec arch("x86_64-pc-windows-msvc");
    Platform::SetHostPlatform(PlatformWindows::CreateInstance(true, &arch));
    m_debugger_sp = Debugger::CreateInstance();
    m_debugger_sp->SetPropertyValue(nullptr,
                                    lldb_private::eVarSetOperationAssign,
                                    "plugin.symbol-file.pdb.reader", "native");
  }

  std::optional<ClangASTMetadata> GetMetadataFor(SymbolFile *symfile,
                                                 llvm::StringRef query,
                                                 TypeSystemClang *clang) {
    TypeResults results;
    symfile->FindTypes(TypeQuery(query), results);
    lldb::TypeSP type_sp = results.GetFirstType();
    if (!type_sp)
      return std::nullopt;

    CompilerType ct = type_sp->GetFullCompilerType();
    ct.GetCompleteType();
    if (!ct.IsValid())
      return std::nullopt;

    ct.IsPossibleDynamicType(nullptr, true, false);

    clang::TagDecl *tag_decl = clang->GetAsTagDecl(ct);
    if (!tag_decl)
      return std::nullopt;

    return clang->GetMetadata(tag_decl);
  }

protected:
  std::string m_test_exe;

  SubsystemRAII<FileSystem, HostInfo, ObjectFilePECOFF, SymbolFileNativePDB,
                SymbolFilePDB, TypeSystemClang>
      m_subsystems;
  lldb::DebuggerSP m_debugger_sp;
};

TEST_F(SymbolFilePDBTests, TestDynamicCxxType) {
  FileSpec fspec(m_test_exe);
  ArchSpec aspec("x86_64-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolFile *symfile = module->GetSymbolFile();
  ASSERT_TRUE(symfile);
  ASSERT_TRUE(llvm::isa<SymbolFileNativePDB>(symfile));

  auto ts = symfile->GetTypeSystemForLanguage(lldb::eLanguageTypeC_plus_plus);
  ASSERT_TRUE(bool(ts));
  TypeSystemClang *clang = llvm::dyn_cast_or_null<TypeSystemClang>(ts->get());
  ASSERT_NE(clang, nullptr);

  auto using_base_meta = GetMetadataFor(symfile, "UsingBase", clang);
  ASSERT_TRUE(using_base_meta.has_value());
  ASSERT_TRUE(using_base_meta->GetIsDynamicCXXType().has_value());
  ASSERT_EQ(using_base_meta->GetIsDynamicCXXType(), true); // has vtable

  auto base_meta = GetMetadataFor(symfile, "Base", clang);
  ASSERT_TRUE(base_meta.has_value());
  ASSERT_TRUE(base_meta->GetIsDynamicCXXType().has_value());
  ASSERT_EQ(base_meta->GetIsDynamicCXXType(), true); // has vtable

  auto vbase_meta = GetMetadataFor(symfile, "VBase", clang);
  ASSERT_TRUE(vbase_meta.has_value());
  ASSERT_TRUE(vbase_meta->GetIsDynamicCXXType().has_value());
  ASSERT_EQ(vbase_meta->GetIsDynamicCXXType(), false); // empty struct

  auto using_vbase_meta = GetMetadataFor(symfile, "UsingVBase", clang);
  ASSERT_TRUE(using_vbase_meta.has_value());
  ASSERT_TRUE(using_vbase_meta->GetIsDynamicCXXType().has_value());
  ASSERT_EQ(using_vbase_meta->GetIsDynamicCXXType(), true); // has virtual base

  auto uu_vbase_meta = GetMetadataFor(symfile, "UsingUsingVBase", clang);
  ASSERT_TRUE(uu_vbase_meta.has_value());
  ASSERT_TRUE(uu_vbase_meta->GetIsDynamicCXXType().has_value());
  ASSERT_EQ(uu_vbase_meta->GetIsDynamicCXXType(),
            true); // has 'UsingVBase' as non-virtual base

  auto not_dynamic_meta = GetMetadataFor(symfile, "NotDynamic", clang);
  ASSERT_TRUE(not_dynamic_meta.has_value());
  ASSERT_TRUE(not_dynamic_meta->GetIsDynamicCXXType().has_value());
  ASSERT_EQ(not_dynamic_meta->GetIsDynamicCXXType(),
            false); // has 'VBase' as non-virtual base
}
