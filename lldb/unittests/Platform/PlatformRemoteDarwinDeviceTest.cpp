//===-- PlatformRemoteDarwinDeviceTest.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/Platform/MacOSX/PlatformRemoteiOS.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"

#include <thread>

using namespace lldb;
using namespace lldb_private;

namespace {
/// A remote-ios platform with a single DeviceSupport directory.
class TestPlatformRemoteiOS : public PlatformRemoteiOS {
public:
  using PlatformRemoteiOS::GetDeviceSupportDirectoryForOSVersion;

  explicit TestPlatformRemoteiOS(const FileSpec &sdk_dir) {
    m_sdk_directory_infos.emplace_back(sdk_dir, sdk_dir.GetFilename());
  }
};
} // namespace

class PlatformRemoteDarwinDeviceTest : public ::testing::Test {
  SubsystemRAII<FileSystem, HostInfo> subsystems;
};

// Target::RebuildModuleListWithExecutable loads dependent modules in parallel,
// and each one can end up in GetDeviceSupportDirectoryForOSVersion. Call it
// from two threads so that ThreadSanitizer can catch unsynchronized accesses
// to the directory it caches.
TEST_F(PlatformRemoteDarwinDeviceTest, DeviceSupportDirectoryForOSVersionRace) {
  FileSpec sdk_dir("/DeviceSupport/18.0 (22A3354)");
  auto platform_sp = std::make_shared<TestPlatformRemoteiOS>(sdk_dir);

  std::string results[2];
  std::thread threads[2];
  for (int i = 0; i < 2; ++i)
    threads[i] = std::thread([&, i] {
      if (const char *dir =
              platform_sp->GetDeviceSupportDirectoryForOSVersion())
        results[i] = dir;
    });
  for (std::thread &thread : threads)
    thread.join();

  EXPECT_EQ(results[0], sdk_dir.GetPath());
  EXPECT_EQ(results[1], sdk_dir.GetPath());
}
