//===-- SymbolFileTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

namespace {

class FakeSymbolFile : public SymbolFileSymtab {
public:
  static void Initialize() {
    PluginManager::RegisterPlugin("SymbolOnlyFakeSymbolFile", "",
                                  CreateSymbolOnlyInstance);
    PluginManager::RegisterPlugin("LineTableFakeSymbolFile", "",
                                  CreateLineTableInstance);
    PluginManager::RegisterPlugin("SymtabLikeFakeSymbolFile", "",
                                  CreateSymtabInstance);
  }

  static void Terminate() {
    PluginManager::UnregisterPlugin(CreateSymtabInstance);
    PluginManager::UnregisterPlugin(CreateLineTableInstance);
    PluginManager::UnregisterPlugin(CreateSymbolOnlyInstance);
  }

  llvm::StringRef GetPluginName() override { return m_plugin_name; }
  uint32_t CalculateAbilities() override { return m_abilities; }

private:
  FakeSymbolFile(ObjectFileSP objfile_sp, llvm::StringRef plugin_name,
                 uint32_t abilities)
      : SymbolFileSymtab(std::move(objfile_sp)), m_plugin_name(plugin_name),
        m_abilities(abilities) {}

  static SymbolFile *CreateSymbolOnlyInstance(ObjectFileSP objfile_sp) {
    return new FakeSymbolFile(std::move(objfile_sp), "SymbolOnlyFakeSymbolFile",
                              Symbols);
  }

  static SymbolFile *CreateSymtabInstance(ObjectFileSP objfile_sp) {
    return new FakeSymbolFile(std::move(objfile_sp), "SymtabLikeFakeSymbolFile",
                              Symbols | CompileUnits);
  }

  static SymbolFile *CreateLineTableInstance(ObjectFileSP objfile_sp) {
    return new FakeSymbolFile(std::move(objfile_sp), "LineTableFakeSymbolFile",
                              CompileUnits | LineTables);
  }

  llvm::StringRef m_plugin_name;
  uint32_t m_abilities;
};

class SymbolFileTest : public testing::Test {
  SubsystemRAII<ObjectFileELF, FakeSymbolFile> subsystems;
};

TEST_F(SymbolFileTest, FindPluginPrefersLineTablesOverSymbols) {
  llvm::Expected<TestFile> file = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
)");
  ASSERT_THAT_EXPECTED(file, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(file->moduleSpec());
  ObjectFile *object_file = module_sp->GetObjectFile();
  ASSERT_NE(object_file, nullptr);

  std::unique_ptr<SymbolFile> symbol_file(
      SymbolFile::FindPlugin(object_file->shared_from_this()));
  ASSERT_NE(symbol_file, nullptr);
  EXPECT_EQ(symbol_file->GetPluginName(), "LineTableFakeSymbolFile");
  EXPECT_EQ(
      symbol_file->GetAbilities(),
      static_cast<uint32_t>(SymbolFile::CompileUnits | SymbolFile::LineTables));
}

} // namespace
