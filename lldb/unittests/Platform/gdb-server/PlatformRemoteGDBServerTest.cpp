//===-- PlatformRemoteGDBServerTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/gdb-server/PlatformRemoteGDBServer.h"
#include "lldb/Utility/Connection.h"
#include "gmock/gmock.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_gdb_server;
using namespace lldb_private::process_gdb_remote;
using namespace testing;

namespace {

class PlatformRemoteGDBServerHack : public PlatformRemoteGDBServer {
public:
  void
  SetGDBClient(std::unique_ptr<GDBRemoteCommunicationClient> gdb_client_up) {
    m_gdb_client_up = std::move(gdb_client_up);
  }
};

class MockConnection : public lldb_private::Connection {
public:
  MOCK_METHOD(lldb::ConnectionStatus, Connect,
              (llvm::StringRef url, Status *error_ptr), (override));
  MOCK_METHOD(lldb::ConnectionStatus, Disconnect, (Status * error_ptr),
              (override));
  MOCK_METHOD(bool, IsConnected, (), (const, override));
  MOCK_METHOD(size_t, Read,
              (void *dst, size_t dst_len, const Timeout<std::micro> &timeout,
               lldb::ConnectionStatus &status, Status *error_ptr),
              (override));
  MOCK_METHOD(size_t, Write,
              (const void *dst, size_t dst_len, lldb::ConnectionStatus &status,
               Status *error_ptr),
              (override));
  MOCK_METHOD(std::string, GetURI, (), (override));
  MOCK_METHOD(bool, InterruptRead, (), (override));
};

} // namespace

TEST(PlatformRemoteGDBServerTest, IsConnected) {
  bool is_connected = true;

  auto connection = std::make_unique<NiceMock<MockConnection>>();
  ON_CALL(*connection, IsConnected())
      .WillByDefault(ReturnPointee(&is_connected));

  auto client = std::make_unique<GDBRemoteCommunicationClient>();
  client->SetConnection(std::move(connection));

  PlatformRemoteGDBServerHack server;
  EXPECT_FALSE(server.IsConnected());

  server.SetGDBClient(std::move(client));
  EXPECT_TRUE(server.IsConnected());

  is_connected = false;
  EXPECT_FALSE(server.IsConnected());
}
