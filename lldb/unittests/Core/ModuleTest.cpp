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
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ConstString.h"
#include "gtest/gtest.h"
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

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

// A caller that supplies a lookup name override is dictating the exact string
// to search the symbol/object files for, so the basename a language plugin
// derives from the user-typed name must not replace it. Language plugins rely
// on this to look up names that are only reachable under their full spelling.
TEST(ModuleTest, MakeLookupInfosHonorsLookupNameOverride) {
  SubsystemRAII<FileSystem, HostInfo, CPlusPlusLanguage> subsystems;

  // "A::count" has the basename "count", which is what an override-less lookup
  // searches for.
  ConstString name("A::count");
  std::vector<Module::LookupInfo> infos = Module::LookupInfo::MakeLookupInfos(
      name, eFunctionNameTypeFull, eLanguageTypeC_plus_plus);
  ASSERT_EQ(infos.size(), 1u);
  EXPECT_EQ(infos[0].GetLookupName(), ConstString("count"));

  // With an override, that exact name must be looked up instead. An override
  // that is textually equal to the user-typed name is still an override.
  for (ConstString override_name : {ConstString("A::count::special"), name}) {
    std::vector<Module::LookupInfo> override_infos =
        Module::LookupInfo::MakeLookupInfos(name, eFunctionNameTypeFull,
                                            eLanguageTypeC_plus_plus,
                                            override_name);
    ASSERT_EQ(override_infos.size(), 1u);
    EXPECT_EQ(override_infos[0].GetLookupName(), override_name);
    // The user-typed name is still what results get filtered against.
    EXPECT_EQ(override_infos[0].GetName(), name);
  }
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

// Module::GetSectionList builds the module's section list lazily. Concurrent
// first-time callers (e.g. AppleObjCRuntime::GetObjCVersion during parallel
// SBTarget module loading) must not race on m_sections_up. This hammers
// GetSectionList from several threads on a fresh module so a sanitizer flags an
// unsynchronized build.
TEST(ModuleTest, GetSectionListConcurrent) {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF, SymbolFileSymtab>
      subsystems;

  // Several sections widen the window during which CreateSections is appending
  // to the SectionList vector while another thread iterates it.
  const char *yaml = R"(
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
  - Name:            .data
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_WRITE ]
    Address:         0x2000
    AddressAlign:    0x10
    Size:            0x100
  - Name:            .rodata
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC ]
    Address:         0x3000
    AddressAlign:    0x10
    Size:            0x100
  - Name:            .bss
    Type:            SHT_NOBITS
    Flags:           [ SHF_ALLOC, SHF_WRITE ]
    Address:         0x4000
    AddressAlign:    0x10
    Size:            0x100
...
)";

  llvm::StringRef text_name(".text");
  constexpr int kThreads = 8;
  // Each iteration uses a fresh module so the lazy build (and its race) is
  // re-triggered every time.
  for (int iter = 0; iter < 100; ++iter) {
    auto ExpectedFile = TestFile::fromYaml(yaml);
    ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
    auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

    // Release the threads together with a blocking gate. A busy-wait would peg
    // every core and starve the workers when many test binaries run at once.
    std::mutex mutex;
    std::condition_variable cv;
    bool go = false;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&] {
        {
          std::unique_lock<std::mutex> lock(mutex);
          cv.wait(lock, [&] { return go; });
        }
        if (SectionList *sections = module_sp->GetSectionList())
          sections->FindSectionByName(text_name);
      });
    }
    {
      std::lock_guard<std::mutex> lock(mutex);
      go = true;
    }
    cv.notify_all();
    for (auto &th : threads)
      th.join();

    // The concurrently-built list must be intact and complete.
    SectionList *sections = module_sp->GetSectionList();
    ASSERT_NE(sections, nullptr);
    EXPECT_TRUE(sections->FindSectionByName(text_name));
  }
}

// ObjectFile::GetSectionList() takes ObjectFile::m_sections_mutex and then
// Module::m_mutex, while callers such as Module::GetUUID() hold Module::m_mutex
// and can reach ObjectFile::GetSectionList() (e.g. ObjectFileELF::GetUUID ->
// GetBaseAddress for in-memory images). Module::SetLoadAddress must take
// Module::m_mutex before calling into the object file, otherwise running it
// concurrently with such a caller (as parallel module loading does) deadlocks.
TEST(ModuleTest, SetLoadAddressConcurrentWithModuleMutexHolder) {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF,
                platform_linux::PlatformLinux>
      subsystems;
  std::call_once(TestUtilities::g_debugger_initialize_flag,
                 []() { Debugger::Initialize(nullptr); });
  ArchSpec arch("x86_64-pc-linux");
  Platform::SetHostPlatform(
      platform_linux::PlatformLinux::CreateInstance(true, &arch));
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);
  TargetSP target_sp;
  PlatformSP platform_sp;
  ASSERT_THAT_ERROR(debugger_sp->GetTargetList()
                        .CreateTarget(*debugger_sp, "", arch, eLoadDependentsNo,
                                      platform_sp, target_sp)
                        .takeError(),
                    llvm::Succeeded());
  ASSERT_TRUE(target_sp);

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
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  // State is shared with the worker threads by value so that, if they
  // deadlock, it outlives this test once they are detached.
  struct State {
    ModuleSP module_sp;
    TargetSP target_sp;
    std::promise<void> holding_module_mutex;
    std::promise<void> setter_started;
    std::promise<void> holder_done;
    std::promise<void> setter_done;
  };
  auto state = std::make_shared<State>();
  state->module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());
  state->target_sp = target_sp;

  // Load the object file up front: the first Module::GetObjectFile() call takes
  // Module::m_mutex. This must not create the section list, as the lazy build
  // in ObjectFile::GetSectionList() is where the lock inversion happens.
  ASSERT_NE(state->module_sp->GetObjectFile(), nullptr);

  // Models Module::GetUUID(): hold Module::m_mutex, then reach
  // ObjectFile::GetSectionList().
  std::thread holder([state] {
    std::lock_guard<std::recursive_mutex> guard(state->module_sp->GetMutex());
    state->holding_module_mutex.set_value();
    state->setter_started.get_future().wait();
    // Give the setter time to get as far as it can. Without the fix it takes
    // ObjectFile::m_sections_mutex and blocks on Module::m_mutex; with the fix
    // it blocks on Module::m_mutex before touching the object file.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    state->module_sp->GetObjectFile()->GetSectionList();
    state->holder_done.set_value();
  });

  std::thread setter([state] {
    state->holding_module_mutex.get_future().wait();
    state->setter_started.set_value();
    bool changed = false;
    state->module_sp->SetLoadAddress(*state->target_sp, 0x10000,
                                     /*value_is_offset=*/true, changed);
    state->setter_done.set_value();
  });

  auto timeout = std::chrono::seconds(30);
  bool finished = state->holder_done.get_future().wait_for(timeout) ==
                      std::future_status::ready &&
                  state->setter_done.get_future().wait_for(timeout) ==
                      std::future_status::ready;
  if (!finished) {
    holder.detach();
    setter.detach();
    FAIL() << "deadlock between Module::SetLoadAddress and a Module::m_mutex "
              "holder calling ObjectFile::GetSectionList";
  }
  holder.join();
  setter.join();
  Debugger::Destroy(debugger_sp);
}
