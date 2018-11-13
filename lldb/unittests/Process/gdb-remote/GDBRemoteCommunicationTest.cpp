//===-- GDBRemoteCommunicationTest.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  TestClient()
      : GDBRemoteCommunication("test.client", "test.client.listener") {}

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
    return server.Write(packet.data(), packet.size(), status, nullptr) ==
           packet.size();
  }
};
} // end anonymous namespace

TEST_F(GDBRemoteCommunicationTest, ReadPacket_checksum) {
  struct TestCase {
    llvm::StringLiteral Packet;
    llvm::StringLiteral Payload;
  };
  static constexpr TestCase Tests[] = {
      {{"$#00"}, {""}},
      {{"$foobar#79"}, {"foobar"}},
      {{"$}}#fa"}, {"]"}},
      {{"$x*%#c7"}, {"xxxxxxxxx"}},
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
