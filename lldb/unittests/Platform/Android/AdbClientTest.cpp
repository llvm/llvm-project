//===-- AdbClientTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/Android/AdbClient.h"
#include "lldb/Host/Socket.h"
#include "lldb/Host/common/TCPSocket.h"
#include "gtest/gtest.h"
#include <chrono>
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
using namespace lldb_private::platform_android;

class AdbClientTest : public ::testing::Test {
public:
  void SetUp() override {
    set_env("ANDROID_SERIAL", "");
    set_env("ANDROID_ADB_SERVER_PORT", "");
  }

  void TearDown() override {
    set_env("ANDROID_SERIAL", "");
    set_env("ANDROID_ADB_SERVER_PORT", "");
  }
};

TEST_F(AdbClientTest, ResolveDeviceId_ExplicitDeviceId) {
  auto result = AdbClient::ResolveDeviceID("device1");
  EXPECT_TRUE(static_cast<bool>(result));
  EXPECT_EQ("device1", *result);
}

TEST_F(AdbClientTest, ResolveDeviceId_ByEnvVar) {
  set_env("ANDROID_SERIAL", "device2");

  auto result = AdbClient::ResolveDeviceID("");
  EXPECT_TRUE(static_cast<bool>(result));
  EXPECT_EQ("device2", *result);
}

TEST_F(AdbClientTest, ResolveDeviceId_PrefersExplicitOverEnvVar) {
  set_env("ANDROID_SERIAL", "env_device");

  // Explicit device ID should take precedence over environment variable
  auto result = AdbClient::ResolveDeviceID("explicit_device");
  EXPECT_TRUE(static_cast<bool>(result));
  EXPECT_EQ("explicit_device", *result);
}

TEST_F(AdbClientTest, AdbClient_Constructor_StoresDeviceId) {
  AdbClient client("test_device_123");
  EXPECT_EQ(client.GetDeviceID(), "test_device_123");
}

TEST_F(AdbClientTest, AdbClient_DefaultConstructor) {
  AdbClient client;
  EXPECT_EQ(client.GetDeviceID(), "");
}

TEST_F(AdbClientTest, AdbSyncService_Constructor_StoresDeviceId) {
  AdbSyncService sync("device123");
  EXPECT_EQ(sync.GetDeviceId(), "device123");
}

TEST_F(AdbClientTest, AdbSyncService_OperationsFailWhenNotConnected) {
  AdbSyncService sync_service("test_device");

  // Verify service is not connected initially
  EXPECT_FALSE(sync_service.IsConnected());

  // File operations should fail when not connected
  FileSpec remote_file("/data/test.txt");
  FileSpec local_file("/tmp/test.txt");
  uint32_t mode, size, mtime;

  Status stat_result = sync_service.Stat(remote_file, mode, size, mtime);
  EXPECT_TRUE(stat_result.Fail());

  Status pull_result = sync_service.PullFile(remote_file, local_file);
  EXPECT_TRUE(pull_result.Fail());

  Status push_result = sync_service.PushFile(local_file, remote_file);
  EXPECT_TRUE(push_result.Fail());
}

static uint16_t FindUnusedPort() {
  auto temp_socket = std::make_unique<TCPSocket>(true);
  Status error = temp_socket->Listen("localhost:0", 1);
  if (error.Fail()) {
    return 0; // fallback
  }
  uint16_t port = temp_socket->GetLocalPortNumber();
  temp_socket.reset(); // Close the socket to free the port
  return port;
}

TEST_F(AdbClientTest, RealTcpConnection) {
  uint16_t unused_port = FindUnusedPort();
  ASSERT_NE(unused_port, 0) << "Failed to find an unused port";

  std::string port_str = std::to_string(unused_port);
  setenv("ANDROID_ADB_SERVER_PORT", port_str.c_str(), 1);

  AdbClient client;
  const auto status1 = client.Connect();
  EXPECT_FALSE(status1.Success())
      << "Connection should fail when no server is listening on port "
      << unused_port;

  // now start a server on the port and try again
  auto listen_socket = std::make_unique<TCPSocket>(true);
  std::string listen_address = "localhost:" + port_str;
  Status error = listen_socket->Listen(listen_address.c_str(), 5);
  ASSERT_TRUE(error.Success()) << "Failed to create listening socket on port "
                               << unused_port << ": " << error.AsCString();

  // Verify the socket is listening on the expected port
  ASSERT_EQ(listen_socket->GetLocalPortNumber(), unused_port)
      << "Socket is not listening on the expected port";

  const auto status2 = client.Connect();
  EXPECT_TRUE(status2.Success())
      << "Connection should succeed when server is listening on port "
      << unused_port;
}
