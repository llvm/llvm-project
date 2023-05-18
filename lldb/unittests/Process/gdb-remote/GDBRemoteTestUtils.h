//===-- GDBRemoteTestUtils.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLDB_UNITTESTS_PROCESS_GDB_REMOTE_GDBREMOTETESTUTILS_H
#define LLDB_UNITTESTS_PROCESS_GDB_REMOTE_GDBREMOTETESTUTILS_H

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServer.h"
#include "lldb/Utility/Connection.h"

namespace lldb_private {
namespace process_gdb_remote {

class GDBRemoteTest : public testing::Test {
public:
  static void SetUpTestCase();
  static void TearDownTestCase();
};

class MockConnection : public lldb_private::Connection {
public:
  MockConnection(std::vector<std::string> &packets) { m_packets = &packets; };

  MOCK_METHOD2(Connect,
               lldb::ConnectionStatus(llvm::StringRef url, Status *error_ptr));
  MOCK_METHOD5(Read, size_t(void *dst, size_t dst_len,
                            const Timeout<std::micro> &timeout,
                            lldb::ConnectionStatus &status, Status *error_ptr));
  MOCK_METHOD0(GetURI, std::string());
  MOCK_METHOD0(InterruptRead, bool());

  lldb::ConnectionStatus Disconnect(Status *error_ptr) {
    return lldb::eConnectionStatusSuccess;
  };

  bool IsConnected() const { return true; };
  size_t Write(const void *dst, size_t dst_len, lldb::ConnectionStatus &status,
               Status *error_ptr) {
    m_packets->emplace_back(static_cast<const char *>(dst), dst_len);
    return dst_len;
  };

  lldb::IOObjectSP GetReadObject() { return lldb::IOObjectSP(); }

  std::vector<std::string> *m_packets;
};

class MockServer : public GDBRemoteCommunicationServer {
public:
  MockServer() : GDBRemoteCommunicationServer() {
    m_send_acks = false;
    m_send_error_strings = true;
  }

  PacketResult SendPacket(llvm::StringRef payload) {
    return GDBRemoteCommunicationServer::SendPacketNoLock(payload);
  }

  PacketResult GetPacket(StringExtractorGDBRemote &response) {
    const bool sync_on_timeout = false;
    return ReadPacket(response, std::chrono::seconds(1), sync_on_timeout);
  }

  using GDBRemoteCommunicationServer::SendErrorResponse;
  using GDBRemoteCommunicationServer::SendOKResponse;
  using GDBRemoteCommunicationServer::SendUnimplementedResponse;
};

class MockServerWithMockConnection : public MockServer {
public:
  MockServerWithMockConnection() : MockServer() {
    SetConnection(std::make_unique<MockConnection>(m_packets));
  }

  llvm::ArrayRef<std::string> GetPackets() { return m_packets; };

  std::vector<std::string> m_packets;
};

} // namespace process_gdb_remote
} // namespace lldb_private

#endif // LLDB_UNITTESTS_PROCESS_GDB_REMOTE_GDBREMOTETESTUTILS_H
