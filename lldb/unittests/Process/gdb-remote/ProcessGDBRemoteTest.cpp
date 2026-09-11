//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Process/gdb-remote/ProcessGDBRemote.h"
#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Diagnostics.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/TargetList.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/Listener.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <mutex>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;

namespace {
class ProcessGDBRemoteDiagnosticsTest : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();
    PlatformMacOSX::Initialize();
    static std::once_flag g_debugger_initialize_flag;
    std::call_once(g_debugger_initialize_flag,
                   [] { Debugger::Initialize(nullptr); });
    Diagnostics::Initialize();
  }
  void TearDown() override {
    Diagnostics::Terminate();
    PlatformMacOSX::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};
} // namespace

// A ProcessGDBRemote registers a packet-history provider while alive and
// unregisters it when destroyed.
TEST_F(ProcessGDBRemoteDiagnosticsTest, PacketHistoryContributesToBundle) {
  ArchSpec arch("x86_64-apple-macosx-");
  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  PlatformSP platform_sp;
  TargetSP target_sp;
  debugger_sp->GetTargetList().CreateTarget(
      *debugger_sp, "", arch, eLoadDependentsNo, platform_sp, target_sp);
  ASSERT_TRUE(target_sp);

  ListenerSP listener_sp = Listener::MakeListener("test-listener");
  ProcessSP process_sp = ProcessGDBRemote::CreateInstance(
      target_sp, listener_sp, /*crash_file_path=*/nullptr,
      /*can_connect=*/true);
  ASSERT_TRUE(process_sp);

  // The provider's file is named with a timestamp, so match it by prefix.
  auto has_packet_history = [](const std::vector<std::string> &files) {
    return llvm::any_of(files, [](llvm::StringRef f) {
      return f.starts_with("gdb-remote-packet-history-");
    });
  };
  ExecutionContext exe_ctx;

  // Alive: the provider contributes the history file.
  {
    llvm::Expected<FileSpec> dir = Diagnostics::CreateUniqueDirectory();
    ASSERT_THAT_EXPECTED(dir, llvm::Succeeded());
    llvm::Expected<Diagnostics::Report> report =
        Diagnostics::Instance().Collect(*debugger_sp, exe_ctx, *dir);
    ASSERT_THAT_EXPECTED(report, llvm::Succeeded());
    EXPECT_TRUE(has_packet_history(report->attachments.files));
    llvm::sys::fs::remove_directories(dir->GetPath());
  }

  // Destroyed: the provider is gone.
  process_sp.reset();
  {
    llvm::Expected<FileSpec> dir = Diagnostics::CreateUniqueDirectory();
    ASSERT_THAT_EXPECTED(dir, llvm::Succeeded());
    llvm::Expected<Diagnostics::Report> report =
        Diagnostics::Instance().Collect(*debugger_sp, exe_ctx, *dir);
    ASSERT_THAT_EXPECTED(report, llvm::Succeeded());
    EXPECT_FALSE(has_packet_history(report->attachments.files));
    llvm::sys::fs::remove_directories(dir->GetPath());
  }
}
