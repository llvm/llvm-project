//===-- PlatformAppleSimulatorTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/Platform/MacOSX/PlatformAppleSimulator.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteAppleTV.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteAppleWatch.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteiOS.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Platform.h"

using namespace lldb;
using namespace lldb_private;

class PlatformAppleSimulatorTest : public ::testing::Test {
  SubsystemRAII<FileSystem, HostInfo, PlatformAppleSimulator, PlatformRemoteiOS,
                PlatformRemoteAppleTV, PlatformRemoteAppleWatch>
      subsystems;
};

#ifdef __APPLE__

static void testSimPlatformArchHasSimEnvironment(llvm::StringRef name) {
  auto platform_sp = Platform::Create(name);
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
    PlatformList list;
    ArchSpec arch = platform_arch;
    arch.GetTriple().setOS(sim);
    arch.GetTriple().setEnvironment(llvm::Triple::Simulator);

    auto platform_sp = list.GetOrCreate(arch, {}, nullptr);
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

  PlatformList list;
  list.GetOrCreate("remote-ios");
  list.GetOrCreate("remote-tvos");
  list.GetOrCreate("remote-watchos");

  for (auto sim : sim_platforms) {
    ArchSpec arch = platform_arch;
    arch.GetTriple().setOS(sim);
    arch.GetTriple().setEnvironment(llvm::Triple::Simulator);

    Status error;
    auto platform_sp = list.GetOrCreate(arch, {}, nullptr, error);
    EXPECT_TRUE(platform_sp);
    EXPECT_TRUE(platform_sp->GetName().contains("simulator"));
  }
}

#endif
