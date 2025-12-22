//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Module.h"
#include "Plugins/Language/CPlusPlus/CPlusPlusLanguage.h"
#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Language.h"
#include "gtest/gtest.h"
#include <optional>

using namespace lldb;
using namespace lldb_private;

// Test that Module::FindFunctions correctly finds C++ mangled symbols
// even when multiple language plugins are registered.
TEST(ModuleTest, FindFunctionsCppMangledName) {
  // Create a mock language. The point of this language is to return something
  // in GetFunctionNameInfo that would interfere with the C++ language plugin,
  // were they sharing the same LookupInfo.
  class MockLanguageWithBogusLookupInfo : public Language {
  public:
    MockLanguageWithBogusLookupInfo() = default;
    ~MockLanguageWithBogusLookupInfo() override = default;

    lldb::LanguageType GetLanguageType() const override {
      // The language here doesn't really matter, it just has to be something
      // that is not C/C++/ObjC.
      return lldb::eLanguageTypeSwift;
    }

    llvm::StringRef GetPluginName() override { return "mock-bogus-language"; }

    bool IsSourceFile(llvm::StringRef file_path) const override {
      return file_path.ends_with(".swift");
    }

    std::pair<lldb::FunctionNameType, std::optional<ConstString>>
    GetFunctionNameInfo(ConstString name) const override {
      // Say that every function is a selector.
      return {lldb::eFunctionNameTypeSelector, ConstString("BOGUS_BASENAME")};
    }

    static void Initialize() {
      PluginManager::RegisterPlugin(GetPluginNameStatic(), "Mock Language",
                                    CreateInstance);
    }

    static void Terminate() { PluginManager::UnregisterPlugin(CreateInstance); }

    static lldb_private::Language *CreateInstance(lldb::LanguageType language) {
      if (language == lldb::eLanguageTypeSwift)
        return new MockLanguageWithBogusLookupInfo();
      return nullptr;
    }

    static llvm::StringRef GetPluginNameStatic() {
      return "mock-bogus-language";
    }
  };
  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF, SymbolFileSymtab,
                CPlusPlusLanguage, MockLanguageWithBogusLookupInfo>
      subsystems;

  // Create a simple ELF module with std::vector::size() as the only symbol.
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x1000
    AddressAlign:    0x10
    Size:            0x100
Symbols:
  - Name:            _ZNSt6vectorIiE4sizeEv
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1030
    Size:            0x20
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  // Verify both C++ and our mock language are registered.
  Language *cpp_lang = Language::FindPlugin(lldb::eLanguageTypeC_plus_plus);
  Language *mock_lang = Language::FindPlugin(lldb::eLanguageTypeSwift);
  ASSERT_NE(cpp_lang, nullptr) << "C++ language plugin should be registered";
  ASSERT_NE(mock_lang, nullptr)
      << "Mock Swift language plugin should be registered";

  ModuleFunctionSearchOptions function_options;
  function_options.include_symbols = true;

  ConstString symbol_name("_ZNSt6vectorIiE4sizeEv");
  SymbolContextList results;
  module_sp->FindFunctions(symbol_name, CompilerDeclContext(),
                           eFunctionNameTypeAuto, function_options, results);

  // Assert that we found one symbol.
  ASSERT_EQ(results.GetSize(), 1u);

  auto result = results[0];
  auto name = result.GetFunctionName();
  // Assert that the symbol we found is what we expected.
  ASSERT_EQ(name, "std::vector<int>::size()");
  ASSERT_EQ(result.GetLanguage(), eLanguageTypeC_plus_plus);
}

TEST(ModuleTest, ResolveSymbolContextForAddressExactMatch) {
  // Test that ResolveSymbolContextForAddress prefers exact symbol matches
  // over symbols that merely contain the address.
  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF, SymbolFileSymtab>
      subsystems;

  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x1000
    AddressAlign:    0x10
    Size:            0x200
Symbols:
  - Name:            outer_function
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1000
    Size:            0x100
  - Name:            inner_function
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1050
    Size:            0x10
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  Address addr(module_sp->GetSectionList()->GetSectionAtIndex(0), 0x50);
  SymbolContext sc;
  uint32_t resolved =
      module_sp->ResolveSymbolContextForAddress(addr, eSymbolContextSymbol, sc);

  ASSERT_TRUE(resolved & eSymbolContextSymbol);
  ASSERT_NE(sc.symbol, nullptr);
  EXPECT_STREQ(sc.symbol->GetName().GetCString(), "inner_function");
}
