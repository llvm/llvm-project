//===-- GDBRemoteTestUtils.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef lldb_unittests_Process_gdb_remote_GDBRemoteTestUtils_h
#define lldb_unittests_Process_gdb_remote_GDBRemoteTestUtils_h

#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServer.h"
#include "gtest/gtest.h"

namespace lldb_private {
namespace process_gdb_remote {

class GDBRemoteTest : public testing::Test {
public:
  static void SetUpTestCase();

  static void TearDownTestCase();
};

void Connect(GDBRemoteCommunication &client, GDBRemoteCommunication &server);

struct MockServer : public GDBRemoteCommunicationServer {
  MockServer()
      : GDBRemoteCommunicationServer("mock-server", "mock-server.listener") {
    m_send_acks = false;
  }

  PacketResult SendPacket(llvm::StringRef payload) {
    return GDBRemoteCommunicationServer::SendPacketNoLock(payload);
  }

  PacketResult GetPacket(StringExtractorGDBRemote &response) {
    const unsigned timeout_usec = 1000000; // 1s
    const bool sync_on_timeout = false;
    return WaitForPacketWithTimeoutMicroSecondsNoLock(response, timeout_usec,
                                                      sync_on_timeout);
  }

  using GDBRemoteCommunicationServer::SendOKResponse;
  using GDBRemoteCommunicationServer::SendUnimplementedResponse;
};

} // namespace process_gdb_remote
} // namespace lldb_private

#endif // lldb_unittests_Process_gdb_remote_GDBRemoteTestUtils_h
