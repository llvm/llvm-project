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

#include <atomic>
#include <thread>
#include <vector>

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

namespace {
class TestSimulatorPlatform : public PlatformAppleSimulator {
public:
  TestSimulatorPlatform()
      : PlatformAppleSimulator(
            "TestSimulatorPlatform", "test", ConstString("test-simulator"),
            llvm::Triple::IOS, {}, "iPhoneSimulator.Internal.sdk",
            "iPhoneSimulator.sdk", XcodeSDK::Type::iPhoneSimulator,
            CoreSimulatorSupport::DeviceType::ProductFamilyID::iPhone) {}

  using PlatformAppleSimulator::GetSDKFilepath;
};
} // namespace

TEST_F(PlatformAppleSimulatorTest, TestConcurrentGetSDKFilepath) {
  constexpr size_t num_threads = 16;
  for (size_t iteration = 0; iteration < 100; ++iteration) {
    TestSimulatorPlatform platform;
    std::atomic<bool> start = false;
    std::vector<std::string> sdks(num_threads);
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads; ++i)
      threads.emplace_back([&, i]() {
        while (!start)
          ;
        sdks[i] = platform.GetSDKFilepath().str();
      });
    start = true;
    for (std::thread &thread : threads)
      thread.join();

    for (const std::string &sdk : sdks)
      EXPECT_EQ(sdk, sdks.front());
  }
}

#endif
