//===---------- SharedModuleList.cpp --------------------------------------===//
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
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Target.h"

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
constexpr llvm::StringLiteral k_arch("aarch64-none-linux");
constexpr llvm::StringLiteral
    k_module_uuid("80008338-82A0-51E5-5922-C905D23890DA-BDDEFECC");
const size_t k_module_size = 3784;

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

ProcessSP MockProcessCreateInstance(TargetSP target_sp, ListenerSP listener_sp,
                                    const FileSpec *crash_file_path,
                                    bool can_connect) {
  return std::make_shared<MockProcess>(target_sp, listener_sp);
}

class SharedModuleListTest : public testing::Test {
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

TEST_F(SharedModuleListTest, TestClear) {
  FileSpec uuid_view = BuildCacheDir(m_test_dir);

  m_module_sp = m_target_sp->GetOrCreateModule(m_module_spec, /*notify=*/false);
  CheckModule(m_module_sp);
  ASSERT_EQ(m_module_sp->GetFileSpec(), uuid_view);
  ASSERT_FALSE(m_module_sp->GetSymbolFileFileSpec());

  UUID uuid = m_module_sp->GetUUID();

  // Check if the module is cached
  ASSERT_TRUE(ModuleList::FindSharedModule(uuid));

  // Clear cache and check that it is gone
  ModuleList::ClearSharedModules();
  ASSERT_FALSE(ModuleList::FindSharedModule(uuid));
  m_module_sp = nullptr;
}
