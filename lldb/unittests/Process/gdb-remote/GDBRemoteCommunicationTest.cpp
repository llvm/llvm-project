//===-- GDBRemoteCommunicationTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "GDBRemoteTestUtils.h"
#include "llvm/Testing/Support/Error.h"

using namespace lldb_private::process_gdb_remote;
using namespace lldb_private;
using namespace lldb;
typedef GDBRemoteCommunication::PacketResult PacketResult;

namespace {

class TestClient : public GDBRemoteCommunication {
public:
  TestClient() : GDBRemoteCommunication() {}

  PacketResult ReadPacket(StringExtractorGDBRemote &response) {
    return GDBRemoteCommunication::ReadPacket(response, std::chrono::seconds(1),
                                              /*sync_on_timeout*/ false);
  }
};

class GDBRemoteCommunicationTest : public GDBRemoteTest {
public:
  void SetUp() override {
    ASSERT_THAT_ERROR(GDBRemoteCommunication::ConnectLocally(client, server),
                      llvm::Succeeded());
  }

protected:
  TestClient client;
  MockServer server;

  bool Write(llvm::StringRef packet) {
    ConnectionStatus status;
    return server.WriteAll(packet.data(), packet.size(), status, nullptr) ==
           packet.size();
  }
};
} // end anonymous namespace

// Test that we can decode packets correctly. In particular, verify that
// checksum calculation works.
TEST_F(GDBRemoteCommunicationTest, ReadPacket) {
  struct TestCase {
    llvm::StringLiteral Packet;
    llvm::StringLiteral Payload;
  };
  static constexpr TestCase Tests[] = {
      {{"$#00"}, {""}},
      {{"$foobar#79"}, {"foobar"}},
      {{"$}]#da"}, {"}"}},          // Escaped }
      {{"$x*%#c7"}, {"xxxxxxxxx"}}, // RLE
      {{"+$#00"}, {""}},            // Spurious ACK
      {{"-$#00"}, {""}},            // Spurious NAK
  };
  for (const auto &Test : Tests) {
    SCOPED_TRACE(Test.Packet + " -> " + Test.Payload);
    StringExtractorGDBRemote response;
    ASSERT_TRUE(Write(Test.Packet));
    ASSERT_EQ(PacketResult::Success, client.ReadPacket(response));
    ASSERT_EQ(Test.Payload, response.GetStringRef());
    ASSERT_EQ(PacketResult::Success, server.GetAck());
  }
}
