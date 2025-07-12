//===-- AdbClientTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/Android/AdbClient.h"
#include "Plugins/Platform/Android/AdbSyncService.h"
#include "gtest/gtest.h"
#include <cstdlib>

static void set_env(const char *var, const char *value) {
#ifdef _WIN32
  _putenv_s(var, value);
#else
  setenv(var, value, true);
#endif
}

using namespace lldb;
using namespace lldb_private;

namespace lldb_private {
namespace platform_android {

class AdbClientTest : public ::testing::Test {
public:
  void SetUp() override { set_env("ANDROID_SERIAL", ""); }

  void TearDown() override { set_env("ANDROID_SERIAL", ""); }
};

TEST_F(AdbClientTest, CreateByDeviceId) {
  AdbClient adb;
  Status error = AdbClient::CreateByDeviceID("device1", adb);
  EXPECT_TRUE(error.Success());
  EXPECT_EQ("device1", adb.GetDeviceID());
}

TEST_F(AdbClientTest, CreateByDeviceId_ByEnvVar) {
  set_env("ANDROID_SERIAL", "device2");

  AdbClient adb;
  Status error = AdbClient::CreateByDeviceID("", adb);
  EXPECT_TRUE(error.Success());
  EXPECT_EQ("device2", adb.GetDeviceID());
}

TEST_F(AdbClientTest, SyncServiceCreation) {
  AdbSyncService sync_service("test_device");
  
  EXPECT_EQ(sync_service.GetDeviceId(), "test_device");
  
  EXPECT_FALSE(sync_service.IsConnected());
}

} // end namespace platform_android
} // end namespace lldb_private
