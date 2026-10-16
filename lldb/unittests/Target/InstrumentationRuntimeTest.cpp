//===-- InstrumentationRuntimeTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/InstrumentationRuntime/BoundsSafety/InstrumentationRuntimeBoundsSafety.h"
#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Breakpoint/BreakpointResolverName.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/Symtab.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

namespace {
class InstrumentationRuntimeTest : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF, SymbolFileSymtab>
      subsystem;
};

constexpr llvm::StringLiteral EmptyImage = R"(
--- !ELF
FileHeader:
  Class: ELFCLASS64
  Data: ELFDATA2LSB
  Type: ET_DYN
  Machine: EM_X86_64
Sections:
  - Name: .text
    Type: SHT_PROGBITS
    Flags: [SHF_ALLOC, SHF_EXECINSTR]
    Address: 0x1000
    Content: C3
)";

} // namespace

TEST_F(InstrumentationRuntimeTest, BoundsSafetyProbeUsesExactCNames) {
  auto image = TestFile::fromYaml(EmptyImage);
  ASSERT_THAT_EXPECTED(image, llvm::Succeeded());

  auto probe = [&](llvm::StringRef name, SymbolType type, bool trampoline,
                   bool synthetic, bool expected, bool annotations = false) {
    SCOPED_TRACE(name.str());
    auto module = std::make_shared<Module>(image->moduleSpec());
    auto sentinel = std::make_shared<Module>(image->moduleSpec());
    ASSERT_NE(module->GetObjectFile(), nullptr);
    ASSERT_NE(sentinel->GetObjectFile(), nullptr);
    module->SetFileSpecAndObjectName(FileSpec("unrelated.so"), ConstString());
    sentinel->SetFileSpecAndObjectName(FileSpec("sentinel.so"), ConstString());
    Symtab *symtab = module->GetObjectFile()->GetSymtab();
    // Give the symtab provider a function even when the candidate is an
    // undefined symbol, so Module::GetSymtab retains the provider.
    symtab->AddSymbol(Symbol(1, "unrelated", eSymbolTypeCode, false, false,
                             false, false, SectionSP(), 0, 0, false, false, 0));
    Symbol symbol(0, name, type, /*external=*/false, /*is_debug=*/false,
                  trampoline, /*is_artificial=*/false, SectionSP(), 0, 0,
                  /*size_is_valid=*/false, annotations, 0);
    // Trampoline classification uses the type, not the legacy constructor flag.
    ASSERT_EQ(symbol.IsTrampoline(), trampoline);
    symbol.SetIsSynthetic(synthetic);
    symtab->AddSymbol(symbol);

    ModuleList modules;
    modules.Append(module);
    modules.Append(sentinel);
    auto runtime = InstrumentationRuntimeBoundsSafety::CreateInstance(nullptr);
    runtime->ModulesDidLoad(modules);

    // A match stops the module walk even though activation has no process.
    // A negative probe reaches the sentinel and creates its raw symbol table.
    EXPECT_EQ(sentinel->GetObjectFile()->GetSymtab(/*can_create=*/false) ==
                  nullptr,
              expected);
    EXPECT_EQ(module->GetSymtabIndexTime().get().count(), 0.0);
  };

  for (auto name : {"__bounds_safety_soft_trap", "__bounds_safety_soft_trap_s"})
    for (auto type :
         {eSymbolTypeCode, eSymbolTypeResolver, eSymbolTypeReExported,
          eSymbolTypeAbsolute, eSymbolTypeUndefined})
      probe(name, type, false, false, true);

  probe("__bounds_safety_soft_trap@VERSION", eSymbolTypeCode, false, false,
        true, true);
  probe("__bounds_safety_soft_trap", eSymbolTypeTrampoline, true, false, false);
  probe("", eSymbolTypeCode, false, true, false);
  probe("_Z25__bounds_safety_soft_trapv", eSymbolTypeCode, false, false, false);
  probe("$s4Test25__bounds_safety_soft_trapyyF", eSymbolTypeCode, false, false,
        false);
}

TEST_F(InstrumentationRuntimeTest, LinkageResolverSerialization) {
  auto resolver = std::make_shared<BreakpointResolverName>(
      nullptr, "runtime", eFunctionNameTypeFull, eLanguageTypeUnknown,
      Breakpoint::Exact, 0, false, false);
  auto serialized = resolver->SerializeToStructuredData();
  auto *dictionary = serialized->GetAsDictionary();
  ASSERT_NE(dictionary, nullptr);
  StructuredData::Dictionary *options = nullptr;
  ASSERT_TRUE(dictionary->GetValueForKeyAsDictionary("Options", options));
  ASSERT_NE(options, nullptr);
  EXPECT_FALSE(options->HasKey("MatchLinkageNames"));

  Status error;
  EXPECT_NE(BreakpointResolver::CreateFromStructuredData(*dictionary, error),
            nullptr);
  EXPECT_TRUE(error.Success());

  resolver->SetMatchLinkageNames(true);
  serialized = resolver->SerializeToStructuredData();
  dictionary = serialized->GetAsDictionary();
  auto restored =
      BreakpointResolver::CreateFromStructuredData(*dictionary, error);
  ASSERT_NE(restored, nullptr);
  auto reserialized = restored->SerializeToStructuredData();
  ASSERT_TRUE(reserialized->GetAsDictionary()->GetValueForKeyAsDictionary(
      "Options", options));
  bool match_linkage_names = false;
  EXPECT_TRUE(options->GetValueForKeyAsBoolean("MatchLinkageNames",
                                               match_linkage_names));
  EXPECT_TRUE(match_linkage_names);

  options->AddStringItem("MatchLinkageNames", "invalid");
  EXPECT_EQ(BreakpointResolverName::CreateFromStructuredData(*options, error),
            nullptr);
  EXPECT_TRUE(error.Fail());
  options->AddBooleanItem("MatchLinkageNames", true);
  options->AddStringItem("Regex", "runtime");
  EXPECT_EQ(BreakpointResolverName::CreateFromStructuredData(*options, error),
            nullptr);
  EXPECT_TRUE(error.Fail());
}
