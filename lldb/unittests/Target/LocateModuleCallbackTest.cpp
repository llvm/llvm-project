//===-- LocateModuleCallbackTest.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/Breakpad/ObjectFileBreakpad.h"
#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/Platform/Android/PlatformAndroid.h"
#include "Plugins/SymbolFile/Breakpad/SymbolFileBreakpad.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Target.h"
#include "gmock/gmock.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_android;
using namespace lldb_private::platform_linux;
using namespace lldb_private::breakpad;
using namespace testing;

namespace {

constexpr llvm::StringLiteral k_process_plugin("mock-process-plugin");
constexpr llvm::StringLiteral k_platform_dir("remote-android");
constexpr llvm::StringLiteral k_cache_dir(".cache");
constexpr llvm::StringLiteral k_module_file("AndroidModule.so");
constexpr llvm::StringLiteral k_symbol_file("AndroidModule.unstripped.so");
constexpr llvm::StringLiteral k_breakpad_symbol_file("AndroidModule.so.sym");
constexpr llvm::StringLiteral k_arch("aarch64-none-linux");
constexpr llvm::StringLiteral
    k_module_uuid("80008338-82A0-51E5-5922-C905D23890DA-BDDEFECC");
constexpr llvm::StringLiteral k_function_symbol("boom");
constexpr llvm::StringLiteral k_hidden_function_symbol("boom_hidden");
const size_t k_module_size = 3784;

ModuleSpec GetTestModuleSpec();

class MockProcess : public Process {
public:
  MockProcess(TargetSP target_sp, ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {}

  llvm::StringRef GetPluginName() override { return k_process_plugin; };

  bool CanDebug(TargetSP target, bool plugin_specified_by_name) override {
    return true;
  }

  Status DoDestroy() override { return Status(); }

  void RefreshStateAfterStop() override {}

  bool DoUpdateThreadList(ThreadList &old_thread_list,
                          ThreadList &new_thread_list) override {
    return false;
  }

  size_t DoReadMemory(addr_t vm_addr, void *buf, size_t size,
                      Status &error) override {
    return 0;
  }

  bool GetModuleSpec(const FileSpec &module_file_spec, const ArchSpec &arch,
                     ModuleSpec &module_spec) override {
    module_spec = GetTestModuleSpec();
    return true;
  }
};

FileSpec GetTestDir() {
  const auto *info = UnitTest::GetInstance()->current_test_info();
  FileSpec test_dir = HostInfo::GetProcessTempDir();
  test_dir.AppendPathComponent(std::string(info->test_case_name()) + "-" +
                               info->name());
  std::error_code ec = llvm::sys::fs::create_directory(test_dir.GetPath());
  EXPECT_FALSE(ec);
  return test_dir;
}

FileSpec GetRemotePath() {
  FileSpec fs("/", FileSpec::Style::posix);
  fs.AppendPathComponent("bin");
  fs.AppendPathComponent(k_module_file);
  return fs;
}

FileSpec GetUuidView(FileSpec spec) {
  spec.AppendPathComponent(k_platform_dir);
  spec.AppendPathComponent(k_cache_dir);
  spec.AppendPathComponent(k_module_uuid);
  spec.AppendPathComponent(k_module_file);
  return spec;
}

void BuildEmptyCacheDir(const FileSpec &test_dir) {
  FileSpec cache_dir(test_dir);
  cache_dir.AppendPathComponent(k_platform_dir);
  cache_dir.AppendPathComponent(k_cache_dir);
  std::error_code ec = llvm::sys::fs::create_directories(cache_dir.GetPath());
  EXPECT_FALSE(ec);
}

FileSpec BuildCacheDir(const FileSpec &test_dir) {
  FileSpec uuid_view = GetUuidView(test_dir);
  std::error_code ec =
      llvm::sys::fs::create_directories(uuid_view.GetDirectory().GetCString());
  EXPECT_FALSE(ec);
  ec = llvm::sys::fs::copy_file(GetInputFilePath(k_module_file),
                                uuid_view.GetPath().c_str());
  EXPECT_FALSE(ec);
  return uuid_view;
}

FileSpec GetSymFileSpec(const FileSpec &uuid_view) {
  return FileSpec(uuid_view.GetPath() + ".sym");
}

FileSpec BuildCacheDirWithSymbol(const FileSpec &test_dir) {
  FileSpec uuid_view = BuildCacheDir(test_dir);
  std::error_code ec =
      llvm::sys::fs::copy_file(GetInputFilePath(k_symbol_file),
                               GetSymFileSpec(uuid_view).GetPath().c_str());
  EXPECT_FALSE(ec);
  return uuid_view;
}

FileSpec BuildCacheDirWithBreakpadSymbol(const FileSpec &test_dir) {
  FileSpec uuid_view = BuildCacheDir(test_dir);
  std::error_code ec =
      llvm::sys::fs::copy_file(GetInputFilePath(k_breakpad_symbol_file),
                               GetSymFileSpec(uuid_view).GetPath().c_str());
  EXPECT_FALSE(ec);
  return uuid_view;
}

ModuleSpec GetTestModuleSpec() {
  ModuleSpec module_spec(GetRemotePath(), ArchSpec(k_arch));
  module_spec.GetUUID().SetFromStringRef(k_module_uuid);
  module_spec.SetObjectSize(k_module_size);
  return module_spec;
}

void CheckModule(const ModuleSP &module_sp) {
  ASSERT_TRUE(module_sp);
  ASSERT_EQ(module_sp->GetUUID().GetAsString(), k_module_uuid);
  ASSERT_EQ(module_sp->GetObjectOffset(), 0U);
  ASSERT_EQ(module_sp->GetPlatformFileSpec(), GetRemotePath());
}

SymbolContextList FindFunctions(const ModuleSP &module_sp,
                                const llvm::StringRef &name) {
  SymbolContextList sc_list;
  ModuleFunctionSearchOptions function_options;
  function_options.include_symbols = true;
  function_options.include_inlines = true;
  FunctionNameType type = static_cast<FunctionNameType>(eSymbolTypeCode);
  module_sp->FindFunctions(ConstString(name), CompilerDeclContext(), type,
                           function_options, sc_list);
  return sc_list;
}

void CheckStrippedSymbol(const ModuleSP &module_sp) {
  SymbolContextList sc_list = FindFunctions(module_sp, k_function_symbol);
  EXPECT_EQ(1U, sc_list.GetSize());

  sc_list = FindFunctions(module_sp, k_hidden_function_symbol);
  EXPECT_EQ(0U, sc_list.GetSize());
}

void CheckUnstrippedSymbol(const ModuleSP &module_sp) {
  SymbolContextList sc_list = FindFunctions(module_sp, k_function_symbol);
  EXPECT_EQ(1U, sc_list.GetSize());

  sc_list = FindFunctions(module_sp, k_hidden_function_symbol);
  EXPECT_EQ(1U, sc_list.GetSize());
}

ProcessSP MockProcessCreateInstance(TargetSP target_sp, ListenerSP listener_sp,
                                    const FileSpec *crash_file_path,
                                    bool can_connect) {
  return std::make_shared<MockProcess>(target_sp, listener_sp);
}

class LocateModuleCallbackTest : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileBreakpad, ObjectFileELF,
                PlatformAndroid, PlatformLinux, SymbolFileBreakpad,
                SymbolFileSymtab>
      subsystems;

public:
  void SetUp() override {
    m_test_dir = GetTestDir();

    // Set module cache directory for PlatformAndroid.
    PlatformAndroid::GetGlobalPlatformProperties().SetModuleCacheDirectory(
        m_test_dir);

    // Create Debugger.
    ArchSpec host_arch("i386-pc-linux");
    Platform::SetHostPlatform(
        platform_linux::PlatformLinux::CreateInstance(true, &host_arch));
    m_debugger_sp = Debugger::CreateInstance();
    EXPECT_TRUE(m_debugger_sp);

    // Create PlatformAndroid.
    ArchSpec arch(k_arch);
    m_platform_sp = PlatformAndroid::CreateInstance(true, &arch);
    EXPECT_TRUE(m_platform_sp);

    // Create Target.
    m_debugger_sp->GetTargetList().CreateTarget(*m_debugger_sp, "", arch,
                                                eLoadDependentsNo,
                                                m_platform_sp, m_target_sp);
    EXPECT_TRUE(m_target_sp);

    // Create MockProcess.
    PluginManager::RegisterPlugin(k_process_plugin, "",
                                  MockProcessCreateInstance);
    m_process_sp =
        m_target_sp->CreateProcess(Listener::MakeListener("test-listener"),
                                   k_process_plugin, /*crash_file=*/nullptr,
                                   /*can_connect=*/true);
    EXPECT_TRUE(m_process_sp);

    m_module_spec = GetTestModuleSpec();
    m_module_spec_without_uuid = ModuleSpec(GetRemotePath(), ArchSpec(k_arch));
  }

  void TearDown() override {
    if (m_module_sp)
      ModuleList::RemoveSharedModule(m_module_sp);
  }

  void CheckNoCallback() {
    EXPECT_FALSE(m_platform_sp->GetLocateModuleCallback());
    EXPECT_EQ(m_callback_call_count, 0);
  }

  void CheckCallbackArgs(const ModuleSpec &module_spec,
                         FileSpec &module_file_spec, FileSpec &symbol_file_spec,
                         const ModuleSpec &expected_module_spec,
                         int expected_callback_call_count) {
    EXPECT_TRUE(expected_module_spec.Matches(module_spec,
                                             /*exact_arch_match=*/true));
    EXPECT_FALSE(module_file_spec);
    EXPECT_FALSE(symbol_file_spec);

    EXPECT_EQ(++m_callback_call_count, expected_callback_call_count);
  }

  void CheckCallbackArgsWithUUID(const ModuleSpec &module_spec,
                                 FileSpec &module_file_spec,
                                 FileSpec &symbol_file_spec,
                                 int expected_callback_call_count) {
    CheckCallbackArgs(module_spec, module_file_spec, symbol_file_spec,
                      m_module_spec, expected_callback_call_count);
    EXPECT_TRUE(module_spec.GetUUID().IsValid());
  }

  void CheckCallbackArgsWithoutUUID(const ModuleSpec &module_spec,
                                    FileSpec &module_file_spec,
                                    FileSpec &symbol_file_spec,
                                    int expected_callback_call_count) {
    CheckCallbackArgs(module_spec, module_file_spec, symbol_file_spec,
                      m_module_spec_without_uuid, expected_callback_call_count);
    EXPECT_FALSE(module_spec.GetUUID().IsValid());
  }

protected:
  FileSpec m_test_dir;
  DebuggerSP m_debugger_sp;
  PlatformSP m_platform_sp;
  TargetSP m_target_sp;
  ProcessSP m_process_sp;
  ModuleSpec m_module_spec;
  ModuleSpec m_module_spec_without_uuid;
  ModuleSP m_module_sp;
  int m_callback_call_count = 0;
};

} // namespace

TEST_F(LocateModuleCallbackTest, GetOrCreateModuleWithCachedModule) {
  // The module file is cached, and the locate module callback is not set.
  // GetOrCreateModule should succeed to return the module from the cache.
  FileSpec uuid_view = BuildCacheDir(m_test_dir);

  CheckNoCallback();

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(), uuid_view);
  ASSERT_FALSE(m_module_sp->GetSymbolFileFileSpec());
  CheckStrippedSymbol(m_module_sp);
}

TEST_F(LocateModuleCallbackTest, GetOrCreateModuleWithCachedModuleAndSymbol) {
  // The module and symbol files are cached, and the locate module callback is
  // not set. GetOrCreateModule should succeed to return the module from the
  // cache with the symbol.
  FileSpec uuid_view = BuildCacheDirWithSymbol(m_test_dir);

  CheckNoCallback();

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(), uuid_view);
  ASSERT_EQ(m_module_sp->GetSymbolFileFileSpec(), GetSymFileSpec(uuid_view));
  CheckUnstrippedSymbol(m_module_sp);
}

TEST_F(LocateModuleCallbackTest,
       GetOrCreateModuleWithCachedModuleAndBreakpadSymbol) {
  // The module file and breakpad symbol file are cached, and the locate module
  // callback is not set. GetOrCreateModule should succeed to return the module
  // from the cache with the symbol.
  FileSpec uuid_view = BuildCacheDirWithBreakpadSymbol(m_test_dir);

  CheckNoCallback();

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(), uuid_view);
  ASSERT_EQ(m_module_sp->GetSymbolFileFileSpec(), GetSymFileSpec(uuid_view));
  CheckUnstrippedSymbol(m_module_sp);
}

TEST_F(LocateModuleCallbackTest, GetOrCreateModuleFailure) {
  // The cache dir is empty, and the locate module callback is not set.
  // GetOrCreateModule should fail because PlatformAndroid tries to download the
  // module and fails.
  BuildEmptyCacheDir(m_test_dir);

  CheckNoCallback();

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  ASSERT_FALSE(m_module_sp);
}

TEST_F(LocateModuleCallbackTest, GetOrCreateModuleCallbackFailureNoCache) {
  // The cache dir is empty, also the locate module callback fails for some
  // reason. GetOrCreateModule should fail because PlatformAndroid tries to
  // download the module and fails.
  BuildEmptyCacheDir(m_test_dir);

  int callback_call_count = 0;
  m_platform_sp->SetLocateModuleCallback(
      [this, &callback_call_count](const ModuleSpec &module_spec,
                                   FileSpec &module_file_spec,
                                   FileSpec &symbol_file_spec) {
        CheckCallbackArgsWithUUID(module_spec, module_file_spec,
                                  symbol_file_spec, ++callback_call_count);
        return Status("The locate module callback failed");
      });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  ASSERT_EQ(callback_call_count, 2);
  ASSERT_FALSE(m_module_sp);
}

TEST_F(LocateModuleCallbackTest, GetOrCreateModuleCallbackFailureCached) {
  // The module file is cached, so GetOrCreateModule should succeed to return
  // the module from the cache even though the locate module callback fails for
  // some reason.
  FileSpec uuid_view = BuildCacheDir(m_test_dir);

  int callback_call_count = 0;
  m_platform_sp->SetLocateModuleCallback(
      [this, &callback_call_count](const ModuleSpec &module_spec,
                                   FileSpec &module_file_spec,
                                   FileSpec &symbol_file_spec) {
        CheckCallbackArgsWithUUID(module_spec, module_file_spec,
                                  symbol_file_spec, ++callback_call_count);
        return Status("The locate module callback failed");
      });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  ASSERT_EQ(callback_call_count, 2);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(), uuid_view);
  ASSERT_FALSE(m_module_sp->GetSymbolFileFileSpec());
  CheckStrippedSymbol(m_module_sp);
}

TEST_F(LocateModuleCallbackTest, GetOrCreateModuleCallbackNoFiles) {
  // The module file is cached, so GetOrCreateModule should succeed to return
  // the module from the cache even though the locate module callback returns
  // no files.
  FileSpec uuid_view = BuildCacheDir(m_test_dir);

  int callback_call_count = 0;
  m_platform_sp->SetLocateModuleCallback(
      [this, &callback_call_count](const ModuleSpec &module_spec,
                                   FileSpec &module_file_spec,
                                   FileSpec &symbol_file_spec) {
        CheckCallbackArgsWithUUID(module_spec, module_file_spec,
                                  symbol_file_spec, ++callback_call_count);
        // The locate module callback succeeds but it does not set
        // module_file_spec nor symbol_file_spec.
        return Status();
      });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  ASSERT_EQ(callback_call_count, 2);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(), uuid_view);
  ASSERT_FALSE(m_module_sp->GetSymbolFileFileSpec());
  CheckStrippedSymbol(m_module_sp);
  ModuleList::RemoveSharedModule(m_module_sp);
}

TEST_F(LocateModuleCallbackTest, GetOrCreateModuleCallbackNonExistentModule) {
  // The module file is cached, so GetOrCreateModule should succeed to return
  // the module from the cache even though the locate module callback returns
  // non-existent module file.
  FileSpec uuid_view = BuildCacheDir(m_test_dir);

  int callback_call_count = 0;
  m_platform_sp->SetLocateModuleCallback(
      [this, &callback_call_count](const ModuleSpec &module_spec,
                                   FileSpec &module_file_spec,
                                   FileSpec &symbol_file_spec) {
        CheckCallbackArgsWithUUID(module_spec, module_file_spec,
                                  symbol_file_spec, ++callback_call_count);
        module_file_spec.SetPath("/this path does not exist");
        return Status();
      });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  ASSERT_EQ(callback_call_count, 2);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(), uuid_view);
  ASSERT_FALSE(m_module_sp->GetSymbolFileFileSpec());
  CheckStrippedSymbol(m_module_sp);
  ModuleList::RemoveSharedModule(m_module_sp);
}

TEST_F(LocateModuleCallbackTest, GetOrCreateModuleCallbackNonExistentSymbol) {
  // The module file is cached, so GetOrCreateModule should succeed to return
  // the module from the cache even though the locate module callback returns
  // non-existent symbol file.
  FileSpec uuid_view = BuildCacheDir(m_test_dir);

  int callback_call_count = 0;
  m_platform_sp->SetLocateModuleCallback(
      [this, &callback_call_count](const ModuleSpec &module_spec,
                                   FileSpec &module_file_spec,
                                   FileSpec &symbol_file_spec) {
        CheckCallbackArgsWithUUID(module_spec, module_file_spec,
                                  symbol_file_spec, ++callback_call_count);
        // The locate module callback returns a right module file.
        module_file_spec.SetPath(GetInputFilePath(k_module_file));
        // But it returns non-existent symbols file.
        symbol_file_spec.SetPath("/this path does not exist");
        return Status();
      });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  ASSERT_EQ(callback_call_count, 2);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(), uuid_view);
  ASSERT_TRUE(m_module_sp->GetSymbolFileFileSpec().GetPath().empty());
  CheckStrippedSymbol(m_module_sp);
  ModuleList::RemoveSharedModule(m_module_sp);
}

TEST_F(LocateModuleCallbackTest, GetOrCreateModuleCallbackSuccessWithModule) {
  // The locate module callback returns a module file, GetOrCreateModule should
  // succeed to return the module from the Inputs directory.
  BuildEmptyCacheDir(m_test_dir);

  m_platform_sp->SetLocateModuleCallback([this](const ModuleSpec &module_spec,
                                                FileSpec &module_file_spec,
                                                FileSpec &symbol_file_spec) {
    CheckCallbackArgsWithUUID(module_spec, module_file_spec, symbol_file_spec,
                              1);
    module_file_spec.SetPath(GetInputFilePath(k_module_file));
    return Status();
  });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(),
            FileSpec(GetInputFilePath(k_module_file)));
  ASSERT_FALSE(m_module_sp->GetSymbolFileFileSpec());
  CheckStrippedSymbol(m_module_sp);
  ModuleList::RemoveSharedModule(m_module_sp);
}

TEST_F(LocateModuleCallbackTest,
       GetOrCreateModuleCallbackSuccessWithSymbolAsModule) {
  // The locate module callback returns the symbol file as a module file. It
  // should work since the sections and UUID of the symbol file are the exact
  // same with the module file, GetOrCreateModule should succeed to return the
  // module with the symbol file from Inputs directory.
  BuildEmptyCacheDir(m_test_dir);

  m_platform_sp->SetLocateModuleCallback([this](const ModuleSpec &module_spec,
                                                FileSpec &module_file_spec,
                                                FileSpec &symbol_file_spec) {
    CheckCallbackArgsWithUUID(module_spec, module_file_spec, symbol_file_spec,
                              1);
    module_file_spec.SetPath(GetInputFilePath(k_symbol_file));
    return Status();
  });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(),
            FileSpec(GetInputFilePath(k_symbol_file)));
  ASSERT_FALSE(m_module_sp->GetSymbolFileFileSpec());
  CheckUnstrippedSymbol(m_module_sp);
  ModuleList::RemoveSharedModule(m_module_sp);
}

TEST_F(LocateModuleCallbackTest,
       GetOrCreateModuleCallbackSuccessWithSymbolAsModuleAndSymbol) {
  // The locate module callback returns a symbol file as both a module file and
  // a symbol file. It should work since the sections and UUID of the symbol
  // file are the exact same with the module file, GetOrCreateModule should
  // succeed to return the module with the symbol file from Inputs directory.
  BuildEmptyCacheDir(m_test_dir);

  m_platform_sp->SetLocateModuleCallback([this](const ModuleSpec &module_spec,
                                                FileSpec &module_file_spec,
                                                FileSpec &symbol_file_spec) {
    CheckCallbackArgsWithUUID(module_spec, module_file_spec, symbol_file_spec,
                              1);
    module_file_spec.SetPath(GetInputFilePath(k_symbol_file));
    symbol_file_spec.SetPath(GetInputFilePath(k_symbol_file));
    return Status();
  });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(),
            FileSpec(GetInputFilePath(k_symbol_file)));
  ASSERT_EQ(m_module_sp->GetSymbolFileFileSpec(),
            FileSpec(GetInputFilePath(k_symbol_file)));
  CheckUnstrippedSymbol(m_module_sp);
  ModuleList::RemoveSharedModule(m_module_sp);
}

TEST_F(LocateModuleCallbackTest,
       GetOrCreateModuleCallbackSuccessWithModuleAndSymbol) {
  // The locate module callback returns a module file and a symbol file,
  // GetOrCreateModule should succeed to return the module from Inputs
  // directory, along with the symbol file.
  BuildEmptyCacheDir(m_test_dir);

  m_platform_sp->SetLocateModuleCallback([this](const ModuleSpec &module_spec,
                                                FileSpec &module_file_spec,
                                                FileSpec &symbol_file_spec) {
    CheckCallbackArgsWithUUID(module_spec, module_file_spec, symbol_file_spec,
                              1);
    module_file_spec.SetPath(GetInputFilePath(k_module_file));
    symbol_file_spec.SetPath(GetInputFilePath(k_symbol_file));
    return Status();
  });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(),
            FileSpec(GetInputFilePath(k_module_file)));
  ASSERT_EQ(m_module_sp->GetSymbolFileFileSpec(),
            FileSpec(GetInputFilePath(k_symbol_file)));
  CheckUnstrippedSymbol(m_module_sp);
  ModuleList::RemoveSharedModule(m_module_sp);
}

TEST_F(LocateModuleCallbackTest,
       GetOrCreateModuleCallbackSuccessWithModuleAndBreakpadSymbol) {
  // The locate module callback returns a module file and a breakpad symbol
  // file, GetOrCreateModule should succeed to return the module with the symbol
  // file from Inputs directory.
  BuildEmptyCacheDir(m_test_dir);

  m_platform_sp->SetLocateModuleCallback([this](const ModuleSpec &module_spec,
                                                FileSpec &module_file_spec,
                                                FileSpec &symbol_file_spec) {
    CheckCallbackArgsWithUUID(module_spec, module_file_spec, symbol_file_spec,
                              1);
    module_file_spec.SetPath(GetInputFilePath(k_module_file));
    symbol_file_spec.SetPath(GetInputFilePath(k_breakpad_symbol_file));
    return Status();
  });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(),
            FileSpec(GetInputFilePath(k_module_file)));
  ASSERT_EQ(m_module_sp->GetSymbolFileFileSpec(),
            FileSpec(GetInputFilePath(k_breakpad_symbol_file)));
  CheckUnstrippedSymbol(m_module_sp);
  ModuleList::RemoveSharedModule(m_module_sp);
}

TEST_F(LocateModuleCallbackTest,
       GetOrCreateModuleCallbackSuccessWithOnlySymbol) {
  // The get callback returns only a symbol file, and the module is cached,
  // GetOrCreateModule should succeed to return the module from the cache
  // along with the symbol file from the Inputs directory.
  FileSpec uuid_view = BuildCacheDir(m_test_dir);

  int callback_call_count = 0;
  m_platform_sp->SetLocateModuleCallback(
      [this, &callback_call_count](const ModuleSpec &module_spec,
                                   FileSpec &module_file_spec,
                                   FileSpec &symbol_file_spec) {
        CheckCallbackArgsWithUUID(module_spec, module_file_spec,
                                  symbol_file_spec, ++callback_call_count);
        symbol_file_spec.SetPath(GetInputFilePath(k_symbol_file));
        return Status();
      });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  ASSERT_EQ(callback_call_count, 2);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(), uuid_view);
  ASSERT_EQ(m_module_sp->GetSymbolFileFileSpec(),
            FileSpec(GetInputFilePath(k_symbol_file)));
  CheckUnstrippedSymbol(m_module_sp);
  ModuleList::RemoveSharedModule(m_module_sp);
}

TEST_F(LocateModuleCallbackTest,
       GetOrCreateModuleCallbackSuccessWithOnlyBreakpadSymbol) {
  // The get callback returns only a breakpad symbol file, and the module is
  // cached, GetOrCreateModule should succeed to return the module from the
  // cache along with the symbol file from the Inputs directory.
  FileSpec uuid_view = BuildCacheDir(m_test_dir);

  int callback_call_count = 0;
  m_platform_sp->SetLocateModuleCallback(
      [this, &callback_call_count](const ModuleSpec &module_spec,
                                   FileSpec &module_file_spec,
                                   FileSpec &symbol_file_spec) {
        CheckCallbackArgsWithUUID(module_spec, module_file_spec,
                                  symbol_file_spec, ++callback_call_count);
        symbol_file_spec.SetPath(GetInputFilePath(k_breakpad_symbol_file));
        return Status();
      });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  ASSERT_EQ(callback_call_count, 2);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(), uuid_view);
  ASSERT_EQ(m_module_sp->GetSymbolFileFileSpec(),
            FileSpec(GetInputFilePath(k_breakpad_symbol_file)));
  CheckUnstrippedSymbol(m_module_sp);
  ModuleList::RemoveSharedModule(m_module_sp);
}

TEST_F(LocateModuleCallbackTest,
       GetOrCreateModuleCallbackSuccessWithMultipleSymbols) {
  // The get callback returns only a symbol file. The first call returns
  // a breakpad symbol file and the second call returns a symbol file.
  // Also the module is cached, so GetOrCreateModule should succeed to return
  // the module from the cache along with the breakpad symbol file from the
  // Inputs directory because GetOrCreateModule will use the first symbol file
  // from the callback.
  FileSpec uuid_view = BuildCacheDir(m_test_dir);

  int callback_call_count = 0;
  m_platform_sp->SetLocateModuleCallback(
      [this, &callback_call_count](const ModuleSpec &module_spec,
                                   FileSpec &module_file_spec,
                                   FileSpec &symbol_file_spec) {
        CheckCallbackArgsWithUUID(module_spec, module_file_spec,
                                  symbol_file_spec, ++callback_call_count);
        symbol_file_spec.SetPath(GetInputFilePath(
            callback_call_count == 1 ? k_breakpad_symbol_file : k_symbol_file));
        return Status();
      });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  ASSERT_EQ(callback_call_count, 2);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(), uuid_view);
  ASSERT_EQ(m_module_sp->GetSymbolFileFileSpec(),
            FileSpec(GetInputFilePath(k_breakpad_symbol_file)));
  CheckUnstrippedSymbol(m_module_sp);
  ModuleList::RemoveSharedModule(m_module_sp);
}

TEST_F(LocateModuleCallbackTest,
       GetOrCreateModuleNoCacheWithCallbackOnlySymbol) {
  // The get callback returns only a symbol file, but the module is not
  // cached, GetOrCreateModule should fail because of the missing module.
  BuildEmptyCacheDir(m_test_dir);

  int callback_call_count = 0;
  m_platform_sp->SetLocateModuleCallback(
      [this, &callback_call_count](const ModuleSpec &module_spec,
                                   FileSpec &module_file_spec,
                                   FileSpec &symbol_file_spec) {
        CheckCallbackArgsWithUUID(module_spec, module_file_spec,
                                  symbol_file_spec, ++callback_call_count);
        symbol_file_spec.SetPath(GetInputFilePath(k_symbol_file));
        return Status();
      });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  ASSERT_EQ(callback_call_count, 2);
  ASSERT_FALSE(m_module_sp);
}

TEST_F(LocateModuleCallbackTest,
       GetOrCreateModuleNoCacheWithCallbackOnlyBreakpadSymbol) {
  // The get callback returns only a breakpad symbol file, but the module is not
  // cached, GetOrCreateModule should fail because of the missing module.
  BuildEmptyCacheDir(m_test_dir);

  int callback_call_count = 0;
  m_platform_sp->SetLocateModuleCallback(
      [this, &callback_call_count](const ModuleSpec &module_spec,
                                   FileSpec &module_file_spec,
                                   FileSpec &symbol_file_spec) {
        CheckCallbackArgsWithUUID(module_spec, module_file_spec,
                                  symbol_file_spec, ++callback_call_count);
        symbol_file_spec.SetPath(GetInputFilePath(k_breakpad_symbol_file));
        return Status();
      });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  ASSERT_EQ(callback_call_count, 2);
  ASSERT_FALSE(m_module_sp);
}

TEST_F(LocateModuleCallbackTest,
       GetOrCreateModuleCallbackSuccessWithModuleByPlatformUUID) {
  // This is a simulation for Android remote platform debugging.
  // The locate module callback first call fails because module_spec does not
  // have UUID. Then, the callback second call returns a module file because the
  // platform resolved the module_spec UUID from the target process.
  // GetOrCreateModule should succeed to return the module from the Inputs
  // directory.
  BuildEmptyCacheDir(m_test_dir);

  int callback_call_count = 0;
  m_platform_sp->SetLocateModuleCallback(
      [this, &callback_call_count](const ModuleSpec &module_spec,
                                   FileSpec &module_file_spec,
                                   FileSpec &symbol_file_spec) {
        callback_call_count++;
        if (callback_call_count == 1) {
          // The module_spec does not have UUID on the first call.
          CheckCallbackArgsWithoutUUID(module_spec, module_file_spec,
                                       symbol_file_spec, callback_call_count);
          return Status("Ignored empty UUID");
        } else {
          // The module_spec has UUID on the second call.
          CheckCallbackArgsWithUUID(module_spec, module_file_spec,
                                    symbol_file_spec, callback_call_count);
          module_file_spec.SetPath(GetInputFilePath(k_module_file));
          return Status();
        }
      });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec_without_uuid,
                                               /*notify=*/false);
  ASSERT_EQ(callback_call_count, 2);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(),
            FileSpec(GetInputFilePath(k_module_file)));
  ASSERT_FALSE(m_module_sp->GetSymbolFileFileSpec());
  CheckStrippedSymbol(m_module_sp);
  ModuleList::RemoveSharedModule(m_module_sp);
}

TEST_F(LocateModuleCallbackTest,
       GetOrCreateModuleCallbackSuccessWithSymbolByPlatformUUID) {
  // Same as GetOrCreateModuleCallbackSuccessWithModuleByPlatformUUID,
  // but with a symbol file. GetOrCreateModule should succeed to return the
  // module file and the symbol file from the Inputs directory.
  BuildEmptyCacheDir(m_test_dir);

  int callback_call_count = 0;
  m_platform_sp->SetLocateModuleCallback(
      [this, &callback_call_count](const ModuleSpec &module_spec,
                                   FileSpec &module_file_spec,
                                   FileSpec &symbol_file_spec) {
        callback_call_count++;
        if (callback_call_count == 1) {
          // The module_spec does not have UUID on the first call.
          CheckCallbackArgsWithoutUUID(module_spec, module_file_spec,
                                       symbol_file_spec, callback_call_count);
          return Status("Ignored empty UUID");
        } else {
          // The module_spec has UUID on the second call.
          CheckCallbackArgsWithUUID(module_spec, module_file_spec,
                                    symbol_file_spec, callback_call_count);
          module_file_spec.SetPath(GetInputFilePath(k_module_file));
          symbol_file_spec.SetPath(GetInputFilePath(k_symbol_file));
          return Status();
        }
      });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec_without_uuid,
                                               /*notify=*/false);
  ASSERT_EQ(callback_call_count, 2);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(),
            FileSpec(GetInputFilePath(k_module_file)));
  ASSERT_EQ(m_module_sp->GetSymbolFileFileSpec(),
            FileSpec(GetInputFilePath(k_symbol_file)));
  CheckUnstrippedSymbol(m_module_sp);
  ModuleList::RemoveSharedModule(m_module_sp);
}

TEST_F(LocateModuleCallbackTest,
       GetOrCreateModuleCallbackSuccessWithBreakpadSymbolByPlatformUUID) {
  // Same as GetOrCreateModuleCallbackSuccessWithModuleByPlatformUUID,
  // but with a breakpad symbol file. GetOrCreateModule should succeed to return
  // the module file and the symbol file from the Inputs directory.
  BuildEmptyCacheDir(m_test_dir);

  int callback_call_count = 0;
  m_platform_sp->SetLocateModuleCallback(
      [this, &callback_call_count](const ModuleSpec &module_spec,
                                   FileSpec &module_file_spec,
                                   FileSpec &symbol_file_spec) {
        callback_call_count++;
        if (callback_call_count == 1) {
          // The module_spec does not have UUID on the first call.
          CheckCallbackArgsWithoutUUID(module_spec, module_file_spec,
                                       symbol_file_spec, callback_call_count);
          return Status("Ignored empty UUID");
        } else {
          // The module_spec has UUID on the second call.
          CheckCallbackArgsWithUUID(module_spec, module_file_spec,
                                    symbol_file_spec, callback_call_count);
          module_file_spec.SetPath(GetInputFilePath(k_module_file));
          symbol_file_spec.SetPath(GetInputFilePath(k_breakpad_symbol_file));
          return Status();
        }
      });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec_without_uuid,
                                               /*notify=*/false);
  ASSERT_EQ(callback_call_count, 2);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(),
            FileSpec(GetInputFilePath(k_module_file)));
  ASSERT_EQ(m_module_sp->GetSymbolFileFileSpec(),
            FileSpec(GetInputFilePath(k_breakpad_symbol_file)));
  CheckUnstrippedSymbol(m_module_sp);
  ModuleList::RemoveSharedModule(m_module_sp);
}

TEST_F(LocateModuleCallbackTest,
       GetOrCreateModuleCallbackSuccessWithOnlyBreakpadSymbolByPlatformUUID) {
  // This is a simulation for Android remote platform debugging.
  // The locate module callback first call fails because module_spec does not
  // have UUID. Then, the callback second call returns a breakpad symbol file
  // for the UUID from the target process. GetOrCreateModule should succeed to
  // return the module from the cache along with the symbol file from the Inputs
  // directory.
  FileSpec uuid_view = BuildCacheDir(m_test_dir);

  int callback_call_count = 0;
  m_platform_sp->SetLocateModuleCallback(
      [this, &callback_call_count](const ModuleSpec &module_spec,
                                   FileSpec &module_file_spec,
                                   FileSpec &symbol_file_spec) {
        callback_call_count++;
        if (callback_call_count == 1) {
          // The module_spec does not have UUID on the first call.
          CheckCallbackArgsWithoutUUID(module_spec, module_file_spec,
                                       symbol_file_spec, callback_call_count);
          return Status("Ignored empty UUID");
        } else {
          // The module_spec has UUID on the second call.
          CheckCallbackArgsWithUUID(module_spec, module_file_spec,
                                    symbol_file_spec, callback_call_count);
          symbol_file_spec.SetPath(GetInputFilePath(k_breakpad_symbol_file));
          return Status();
        }
      });

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec_without_uuid,
                                               /*notify=*/false);
  ASSERT_EQ(callback_call_count, 2);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(), uuid_view);
  ASSERT_EQ(m_module_sp->GetSymbolFileFileSpec(),
            FileSpec(GetInputFilePath(k_breakpad_symbol_file)));
  CheckUnstrippedSymbol(m_module_sp);
  ModuleList::RemoveSharedModule(m_module_sp);
}
