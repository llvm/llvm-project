//===-- GDBRemoteTestUtils.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef lldb_unittests_Process_gdb_remote_GDBRemoteTestUtils_h
#define lldb_unittests_Process_gdb_remote_GDBRemoteTestUtils_h

#include "gtest/gtest.h"

#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServer.h"

namespace lldb_private {
namespace process_gdb_remote {

class GDBRemoteTest : public testing::Test {
public:
  static void SetUpTestCase();
  static void TearDownTestCase();
};

struct MockServer : public GDBRemoteCommunicationServer {
  MockServer()
      : GDBRemoteCommunicationServer("mock-server", "mock-server.listener") {
    m_send_acks = false;
  }

  PacketResult SendPacket(llvm::StringRef payload) {
    return GDBRemoteCommunicationServer::SendPacketNoLock(payload);
  }

  PacketResult GetPacket(StringExtractorGDBRemote &response) {
    const bool sync_on_timeout = false;
    return WaitForPacketNoLock(response, std::chrono::seconds(1),
                               sync_on_timeout);
  }

  using GDBRemoteCommunicationServer::SendOKResponse;
  using GDBRemoteCommunicationServer::SendUnimplementedResponse;
};

} // namespace process_gdb_remote
} // namespace lldb_private

#endif // lldb_unittests_Process_gdb_remote_GDBRemoteTestUtils_h
