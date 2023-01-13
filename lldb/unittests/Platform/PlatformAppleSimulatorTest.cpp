//===-- PlatformAppleSimulatorTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/Platform/MacOSX/PlatformAppleSimulator.h"
#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteAppleTV.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteAppleWatch.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteiOS.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Platform.h"

using namespace lldb;
using namespace lldb_private;

static std::once_flag platform_initialize_flag;
static std::once_flag debugger_initialize_flag;

class PlatformAppleSimulatorTest : public ::testing::Test {
  SubsystemRAII<FileSystem, HostInfo, PlatformAppleSimulator, PlatformRemoteiOS,
                PlatformRemoteAppleTV, PlatformRemoteAppleWatch>
      subsystems;

public:
  void SetUp() override {
    std::call_once(platform_initialize_flag,
                   []() { PlatformMacOSX::Initialize(); });
    std::call_once(debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });
    ArchSpec arch("x86_64-apple-macosx-");
    Platform::SetHostPlatform(
        PlatformRemoteMacOSX::CreateInstance(true, &arch, /*debugger=*/nullptr,
                                             /*metadata=*/nullptr));
  }
  void TearDown() override { PlatformMacOSX::Terminate(); }

protected:
  DebuggerSP m_debugger_sp = nullptr;
};

#ifdef __APPLE__

static void testSimPlatformArchHasSimEnvironment(llvm::StringRef name) {
  auto platform_sp = Platform::Create(name, /*debugger=*/nullptr,
                                      /*metadata=*/nullptr);
  ASSERT_TRUE(platform_sp);
  int num_arches = 0;

  for (auto arch : platform_sp->GetSupportedArchitectures({})) {
    EXPECT_EQ(arch.GetTriple().getEnvironment(), llvm::Triple::Simulator);
    num_arches++;
  }

  EXPECT_GT(num_arches, 0);
}

TEST_F(PlatformAppleSimulatorTest, TestSimHasSimEnvionament) {
  testSimPlatformArchHasSimEnvironment("ios-simulator");
  testSimPlatformArchHasSimEnvironment("tvos-simulator");
  testSimPlatformArchHasSimEnvironment("watchos-simulator");
}

TEST_F(PlatformAppleSimulatorTest, TestHostPlatformToSim) {
  static const ArchSpec platform_arch(
      HostInfo::GetArchitecture(HostInfo::eArchKindDefault));

  const llvm::Triple::OSType sim_platforms[] = {
      llvm::Triple::IOS,
      llvm::Triple::TvOS,
      llvm::Triple::WatchOS,
  };

  for (auto sim : sim_platforms) {
    PlatformList list(*m_debugger_sp.get());
    ArchSpec arch = platform_arch;
    arch.GetTriple().setOS(sim);
    arch.GetTriple().setEnvironment(llvm::Triple::Simulator);

    auto platform_sp = list.GetOrCreate(arch, {}, /*platform_arch_ptr=*/nullptr,
                                        /*metadata=*/nullptr);
    EXPECT_TRUE(platform_sp);
  }
}

TEST_F(PlatformAppleSimulatorTest, TestPlatformSelectionOrder) {
  static const ArchSpec platform_arch(
      HostInfo::GetArchitecture(HostInfo::eArchKindDefault));

  const llvm::Triple::OSType sim_platforms[] = {
      llvm::Triple::IOS,
      llvm::Triple::TvOS,
      llvm::Triple::WatchOS,
  };

  PlatformList list(*m_debugger_sp.get());
  list.GetOrCreate("remote-ios", /*metadata=*/nullptr);
  list.GetOrCreate("remote-tvos", /*metadata=*/nullptr);
  list.GetOrCreate("remote-watchos", /*metadata=*/nullptr);

  for (auto sim : sim_platforms) {
    ArchSpec arch = platform_arch;
    arch.GetTriple().setOS(sim);
    arch.GetTriple().setEnvironment(llvm::Triple::Simulator);

    Status error;
    auto platform_sp =
        list.GetOrCreate(arch, {}, /*platform_arch_ptr=*/nullptr, error,
                         /*metadata=*/nullptr);
    EXPECT_TRUE(platform_sp);
    EXPECT_TRUE(platform_sp->GetName().contains("simulator"));
  }
}

#endif
