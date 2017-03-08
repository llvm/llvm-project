//===-- GDBRemoteCommunicationClientTest.cpp --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <future>

#include "GDBRemoteTestUtils.h"

#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationClient.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/StructuredData.h"
#include "lldb/Utility/DataBuffer.h"

#include "llvm/ADT/ArrayRef.h"

using namespace lldb_private::process_gdb_remote;
using namespace lldb_private;
using namespace lldb;
using namespace llvm;

namespace {

typedef GDBRemoteCommunication::PacketResult PacketResult;

struct TestClient : public GDBRemoteCommunicationClient {
  TestClient() { m_send_acks = false; }
};

void Handle_QThreadSuffixSupported(MockServer &server, bool supported) {
  StringExtractorGDBRemote request;
  ASSERT_EQ(PacketResult::Success, server.GetPacket(request));
  ASSERT_EQ("QThreadSuffixSupported", request.GetStringRef());
  if (supported)
    ASSERT_EQ(PacketResult::Success, server.SendOKResponse());
  else
    ASSERT_EQ(PacketResult::Success, server.SendUnimplementedResponse(nullptr));
}

void HandlePacket(MockServer &server, StringRef expected, StringRef response) {
  StringExtractorGDBRemote request;
  ASSERT_EQ(PacketResult::Success, server.GetPacket(request));
  ASSERT_EQ(expected, request.GetStringRef());
  ASSERT_EQ(PacketResult::Success, server.SendPacket(response));
}

uint8_t all_registers[] = {'@', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                           'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O'};
std::string all_registers_hex = "404142434445464748494a4b4c4d4e4f";
uint8_t one_register[] = {'A', 'B', 'C', 'D'};
std::string one_register_hex = "41424344";

} // end anonymous namespace

class GDBRemoteCommunicationClientTest : public GDBRemoteTest {};

TEST_F(GDBRemoteCommunicationClientTest, WriteRegister) {
  TestClient client;
  MockServer server;
  Connect(client, server);
  if (HasFailure())
    return;

  const lldb::tid_t tid = 0x47;
  const uint32_t reg_num = 4;
  std::future<bool> write_result = std::async(std::launch::async, [&] {
    return client.WriteRegister(tid, reg_num, one_register);
  });

  Handle_QThreadSuffixSupported(server, true);

  HandlePacket(server, "P4=" + one_register_hex + ";thread:0047;", "OK");
  ASSERT_TRUE(write_result.get());

  write_result = std::async(std::launch::async, [&] {
    return client.WriteAllRegisters(tid, all_registers);
  });

  HandlePacket(server, "G" + all_registers_hex + ";thread:0047;", "OK");
  ASSERT_TRUE(write_result.get());
}

TEST_F(GDBRemoteCommunicationClientTest, WriteRegisterNoSuffix) {
  TestClient client;
  MockServer server;
  Connect(client, server);
  if (HasFailure())
    return;

  const lldb::tid_t tid = 0x47;
  const uint32_t reg_num = 4;
  std::future<bool> write_result = std::async(std::launch::async, [&] {
    return client.WriteRegister(tid, reg_num, one_register);
  });

  Handle_QThreadSuffixSupported(server, false);
  HandlePacket(server, "Hg47", "OK");
  HandlePacket(server, "P4=" + one_register_hex, "OK");
  ASSERT_TRUE(write_result.get());

  write_result = std::async(std::launch::async, [&] {
    return client.WriteAllRegisters(tid, all_registers);
  });

  HandlePacket(server, "G" + all_registers_hex, "OK");
  ASSERT_TRUE(write_result.get());
}

TEST_F(GDBRemoteCommunicationClientTest, ReadRegister) {
  TestClient client;
  MockServer server;
  Connect(client, server);
  if (HasFailure())
    return;

  const lldb::tid_t tid = 0x47;
  const uint32_t reg_num = 4;
  std::future<bool> async_result = std::async(
      std::launch::async, [&] { return client.GetpPacketSupported(tid); });
  Handle_QThreadSuffixSupported(server, true);
  HandlePacket(server, "p0;thread:0047;", one_register_hex);
  ASSERT_TRUE(async_result.get());

  std::future<DataBufferSP> read_result = std::async(
      std::launch::async, [&] { return client.ReadRegister(tid, reg_num); });
  HandlePacket(server, "p4;thread:0047;", "41424344");
  auto buffer_sp = read_result.get();
  ASSERT_TRUE(bool(buffer_sp));
  ASSERT_EQ(0,
            memcmp(buffer_sp->GetBytes(), one_register, sizeof one_register));

  read_result = std::async(std::launch::async,
                           [&] { return client.ReadAllRegisters(tid); });
  HandlePacket(server, "g;thread:0047;", all_registers_hex);
  buffer_sp = read_result.get();
  ASSERT_TRUE(bool(buffer_sp));
  ASSERT_EQ(0,
            memcmp(buffer_sp->GetBytes(), all_registers, sizeof all_registers));
}

TEST_F(GDBRemoteCommunicationClientTest, SaveRestoreRegistersNoSuffix) {
  TestClient client;
  MockServer server;
  Connect(client, server);
  if (HasFailure())
    return;

  const lldb::tid_t tid = 0x47;
  uint32_t save_id;
  std::future<bool> async_result = std::async(std::launch::async, [&] {
    return client.SaveRegisterState(tid, save_id);
  });
  Handle_QThreadSuffixSupported(server, false);
  HandlePacket(server, "Hg47", "OK");
  HandlePacket(server, "QSaveRegisterState", "1");
  ASSERT_TRUE(async_result.get());
  EXPECT_EQ(1u, save_id);

  async_result = std::async(std::launch::async, [&] {
    return client.RestoreRegisterState(tid, save_id);
  });
  HandlePacket(server, "QRestoreRegisterState:1", "OK");
  ASSERT_TRUE(async_result.get());
}

TEST_F(GDBRemoteCommunicationClientTest, SyncThreadState) {
  TestClient client;
  MockServer server;
  Connect(client, server);
  if (HasFailure())
    return;

  const lldb::tid_t tid = 0x47;
  std::future<bool> async_result = std::async(
      std::launch::async, [&] { return client.SyncThreadState(tid); });
  HandlePacket(server, "qSyncThreadStateSupported", "OK");
  HandlePacket(server, "QSyncThreadState:0047;", "OK");
  ASSERT_TRUE(async_result.get());
}

TEST_F(GDBRemoteCommunicationClientTest, GetModulesInfo) {
  TestClient client;
  MockServer server;
  Connect(client, server);
  if (HasFailure())
    return;

  llvm::Triple triple("i386-pc-linux");

  FileSpec file_specs[] = {
      FileSpec("/foo/bar.so", false, FileSpec::ePathSyntaxPosix),
      FileSpec("/foo/baz.so", false, FileSpec::ePathSyntaxPosix),

      // This is a bit dodgy but we currently depend on GetModulesInfo not
      // performing denormalization. It can go away once the users
      // (DynamicLoaderPOSIXDYLD, at least) correctly set the path syntax for
      // the FileSpecs they create.
      FileSpec("/foo/baw.so", false, FileSpec::ePathSyntaxWindows),
  };
  std::future<llvm::Optional<std::vector<ModuleSpec>>> async_result =
      std::async(std::launch::async,
                 [&] { return client.GetModulesInfo(file_specs, triple); });
  HandlePacket(
      server, "jModulesInfo:["
              R"({"file":"/foo/bar.so","triple":"i386-pc-linux"},)"
              R"({"file":"/foo/baz.so","triple":"i386-pc-linux"},)"
              R"({"file":"/foo/baw.so","triple":"i386-pc-linux"}])",
      R"([{"uuid":"404142434445464748494a4b4c4d4e4f","triple":"i386-pc-linux",)"
      R"("file_path":"/foo/bar.so","file_offset":0,"file_size":1234}]])");

  auto result = async_result.get();
  ASSERT_TRUE(result.hasValue());
  ASSERT_EQ(1u, result->size());
  EXPECT_EQ("/foo/bar.so", result.getValue()[0].GetFileSpec().GetPath());
  EXPECT_EQ(triple, result.getValue()[0].GetArchitecture().GetTriple());
  EXPECT_EQ(UUID("@ABCDEFGHIJKLMNO", 16), result.getValue()[0].GetUUID());
  EXPECT_EQ(0u, result.getValue()[0].GetObjectOffset());
  EXPECT_EQ(1234u, result.getValue()[0].GetObjectSize());
}

TEST_F(GDBRemoteCommunicationClientTest, GetModulesInfoInvalidResponse) {
  TestClient client;
  MockServer server;
  Connect(client, server);
  if (HasFailure())
    return;

  llvm::Triple triple("i386-pc-linux");
  FileSpec file_spec("/foo/bar.so", false, FileSpec::ePathSyntaxPosix);

  const char *invalid_responses[] = {
      "OK", "E47", "[]",
      // no UUID
      R"([{"triple":"i386-pc-linux",)"
      R"("file_path":"/foo/bar.so","file_offset":0,"file_size":1234}])",
      // no triple
      R"([{"uuid":"404142434445464748494a4b4c4d4e4f",)"
      R"("file_path":"/foo/bar.so","file_offset":0,"file_size":1234}])",
      // no file_path
      R"([{"uuid":"404142434445464748494a4b4c4d4e4f","triple":"i386-pc-linux",)"
      R"("file_offset":0,"file_size":1234}])",
      // no file_offset
      R"([{"uuid":"404142434445464748494a4b4c4d4e4f","triple":"i386-pc-linux",)"
      R"("file_path":"/foo/bar.so","file_size":1234}])",
      // no file_size
      R"([{"uuid":"404142434445464748494a4b4c4d4e4f","triple":"i386-pc-linux",)"
      R"("file_path":"/foo/bar.so","file_offset":0}])",
  };

  for (const char *response : invalid_responses) {
    std::future<llvm::Optional<std::vector<ModuleSpec>>> async_result =
        std::async(std::launch::async,
                   [&] { return client.GetModulesInfo(file_spec, triple); });
    HandlePacket(
        server,
        R"(jModulesInfo:[{"file":"/foo/bar.so","triple":"i386-pc-linux"}])",
        response);

    ASSERT_FALSE(async_result.get().hasValue()) << "response was: " << response;
  }
}

TEST_F(GDBRemoteCommunicationClientTest, TestPacketSpeedJSON) {
  TestClient client;
  MockServer server;
  Connect(client, server);
  if (HasFailure())
    return;

  std::thread server_thread([&server] {
    for (;;) {
      StringExtractorGDBRemote request;
      PacketResult result = server.GetPacket(request);
      if (result == PacketResult::ErrorDisconnected)
        return;
      ASSERT_EQ(PacketResult::Success, result);
      StringRef ref = request.GetStringRef();
      ASSERT_TRUE(ref.consume_front("qSpeedTest:response_size:"));
      int size;
      ASSERT_FALSE(ref.consumeInteger(10, size)) << "ref: " << ref;
      std::string response(size, 'X');
      ASSERT_EQ(PacketResult::Success, server.SendPacket(response));
    }
  });

  StreamString ss;
  client.TestPacketSpeed(10, 32, 32, 4096, true, ss);
  client.Disconnect();
  server_thread.join();

  GTEST_LOG_(INFO) << "Formatted output: " << ss.GetData();
  auto object_sp = StructuredData::ParseJSON(ss.GetString());
  ASSERT_TRUE(bool(object_sp));
  auto dict_sp = object_sp->GetAsDictionary();
  ASSERT_TRUE(bool(dict_sp));

  object_sp = dict_sp->GetValueForKey("packet_speeds");
  ASSERT_TRUE(bool(object_sp));
  dict_sp = object_sp->GetAsDictionary();
  ASSERT_TRUE(bool(dict_sp));

  int num_packets;
  ASSERT_TRUE(dict_sp->GetValueForKeyAsInteger("num_packets", num_packets))
      << ss.GetString();
  ASSERT_EQ(10, num_packets);
}

TEST_F(GDBRemoteCommunicationClientTest, SendSignalsToIgnore) {
  TestClient client;
  MockServer server;
  Connect(client, server);
  if (HasFailure())
    return;

  const lldb::tid_t tid = 0x47;
  const uint32_t reg_num = 4;
  std::future<Error> result = std::async(std::launch::async, [&] {
    return client.SendSignalsToIgnore({2, 3, 5, 7, 0xB, 0xD, 0x11});
  });

  HandlePacket(server, "QPassSignals:02;03;05;07;0b;0d;11", "OK");
  EXPECT_TRUE(result.get().Success());

  result = std::async(std::launch::async, [&] {
    return client.SendSignalsToIgnore(std::vector<int32_t>());
  });

  HandlePacket(server, "QPassSignals:", "OK");
  EXPECT_TRUE(result.get().Success());
}
