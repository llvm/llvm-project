//===-- PlatformAppleSimulatorTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/Platform/MacOSX/PlatformAppleTVSimulator.h"
#include "Plugins/Platform/MacOSX/PlatformAppleWatchSimulator.h"
#include "Plugins/Platform/MacOSX/PlatformiOSSimulator.h"
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
  SubsystemRAII<FileSystem, HostInfo, PlatformAppleTVSimulator,
                PlatformiOSSimulator, PlatformAppleWatchSimulator,
                PlatformRemoteAppleTV, PlatformRemoteAppleWatch,
                PlatformRemoteiOS>
      subsystems;
};

#ifdef __APPLE__

static void testSimPlatformArchHasSimEnvironment(llvm::StringRef name) {
  Status error;
  auto platform_sp = Platform::Create(ConstString(name), error);
  ASSERT_TRUE(platform_sp);
  int num_arches = 0;

  while (true) {
    ArchSpec arch;
    if (!platform_sp->GetSupportedArchitectureAtIndex(num_arches, arch))
      break;
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
    ArchSpec arch = platform_arch;
    arch.GetTriple().setOS(sim);
    arch.GetTriple().setEnvironment(llvm::Triple::Simulator);

    Status error;
    auto platform_sp = Platform::Create(arch, nullptr, error);
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

  Status error;
  Platform::Create(ConstString("remote-ios"), error);
  EXPECT_FALSE(error.Fail());
  Platform::Create(ConstString("remote-tvos"), error);
  EXPECT_FALSE(error.Fail());
  Platform::Create(ConstString("remote-watchos"), error);    
  EXPECT_FALSE(error.Fail());

  for (auto sim : sim_platforms) {
    ArchSpec arch = platform_arch;
    arch.GetTriple().setOS(sim);
    arch.GetTriple().setEnvironment(llvm::Triple::Simulator);

    auto platform_sp = Platform::Create(arch, nullptr, error);
    EXPECT_TRUE(platform_sp->GetName().GetStringRef().contains("simulator"));
  }
}


#endif
