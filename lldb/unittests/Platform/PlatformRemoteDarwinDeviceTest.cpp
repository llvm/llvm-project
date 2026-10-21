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

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace lldb;
using namespace lldb_private;

namespace {
/// A remote-ios platform with a fixed list of DeviceSupport directories that
/// counts how many threads are filling the
/// GetDeviceSupportDirectoryForOSVersion cache at the same time.
class TestPlatformRemoteiOS : public PlatformRemoteiOS {
public:
  using PlatformRemoteiOS::GetDeviceSupportDirectoryForOSVersion;

  explicit TestPlatformRemoteiOS(const FileSpec &sdk_dir) {
    m_sdk_directory_infos.emplace_back(sdk_dir, sdk_dir.GetFilename());
  }

  unsigned GetMaxFillers() {
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_max_fillers;
  }

protected:
  // Only called while the cache is being filled. Wait here for a while to give
  // a second thread the chance to get past the "is the cache empty" check too.
  std::optional<std::string> GetRemoteOSBuildString() override {
    std::unique_lock<std::mutex> lock(m_mutex);
    ++m_fillers;
    m_max_fillers = std::max(m_max_fillers, m_fillers);
    m_cv.notify_all();
    m_cv.wait_for(lock, std::chrono::seconds(1),
                  [this] { return m_max_fillers > 1; });
    --m_fillers;
    return std::nullopt;
  }

private:
  std::mutex m_mutex;
  std::condition_variable m_cv;
  unsigned m_fillers = 0;
  unsigned m_max_fillers = 0;
};
} // namespace

class PlatformRemoteDarwinDeviceTest : public ::testing::Test {
  SubsystemRAII<FileSystem, HostInfo> subsystems;
};

// Target::SetExecutableModule loads dependent modules in parallel, and each
// one ends up in GetDeviceSupportDirectoryForOSVersion. The cache it fills
// must not be written by more than one thread at a time.
TEST_F(PlatformRemoteDarwinDeviceTest, DeviceSupportDirectoryForOSVersionRace) {
  // Long enough to not fit in std::string's small buffer, like the real
  // ~/Library/Developer/Xcode/iOS DeviceSupport/... paths.
  FileSpec sdk_dir("/Users/someone/Library/Developer/Xcode/iOS DeviceSupport/"
                   "iPhone17,1 18.0 (22A3354)");
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

  EXPECT_EQ(platform_sp->GetMaxFillers(), 1u);
  EXPECT_EQ(results[0], sdk_dir.GetPath());
  EXPECT_EQ(results[1], sdk_dir.GetPath());
}
