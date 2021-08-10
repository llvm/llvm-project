//===-- GDBRemoteCommunicationClientTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationClient.h"
#include "GDBRemoteTestUtils.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/XML.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include <future>
#include <limits>

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

void HandlePacket(MockServer &server,
                  const testing::Matcher<const std::string &> &expected,
                  StringRef response) {
  StringExtractorGDBRemote request;
  ASSERT_EQ(PacketResult::Success, server.GetPacket(request));
  ASSERT_THAT(std::string(request.GetStringRef()), expected);
  ASSERT_EQ(PacketResult::Success, server.SendPacket(response));
}

uint8_t all_registers[] = {'@', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                           'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O'};
std::string all_registers_hex = "404142434445464748494a4b4c4d4e4f";
uint8_t one_register[] = {'A', 'B', 'C', 'D'};
std::string one_register_hex = "41424344";

} // end anonymous namespace

class GDBRemoteCommunicationClientTest : public GDBRemoteTest {
public:
  void SetUp() override {
    ASSERT_THAT_ERROR(GDBRemoteCommunication::ConnectLocally(client, server),
                      llvm::Succeeded());
  }

protected:
  TestClient client;
  MockServer server;
};

TEST_F(GDBRemoteCommunicationClientTest, WriteRegister) {
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
  const lldb::tid_t tid = 0x47;
  std::future<bool> async_result = std::async(
      std::launch::async, [&] { return client.SyncThreadState(tid); });
  HandlePacket(server, "qSyncThreadStateSupported", "OK");
  HandlePacket(server, "QSyncThreadState:0047;", "OK");
  ASSERT_TRUE(async_result.get());
}

TEST_F(GDBRemoteCommunicationClientTest, GetModulesInfo) {
  llvm::Triple triple("i386-pc-linux");

  FileSpec file_specs[] = {
      FileSpec("/foo/bar.so", FileSpec::Style::posix),
      FileSpec("/foo/baz.so", FileSpec::Style::posix),

      // This is a bit dodgy but we currently depend on GetModulesInfo not
      // performing denormalization. It can go away once the users
      // (DynamicLoaderPOSIXDYLD, at least) correctly set the path syntax for
      // the FileSpecs they create.
      FileSpec("/foo/baw.so", FileSpec::Style::windows),
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
  EXPECT_EQ(UUID::fromData("@ABCDEFGHIJKLMNO", 16),
            result.getValue()[0].GetUUID());
  EXPECT_EQ(0u, result.getValue()[0].GetObjectOffset());
  EXPECT_EQ(1234u, result.getValue()[0].GetObjectSize());
}

TEST_F(GDBRemoteCommunicationClientTest, GetModulesInfo_UUID20) {
  llvm::Triple triple("i386-pc-linux");

  FileSpec file_spec("/foo/bar.so", FileSpec::Style::posix);
  std::future<llvm::Optional<std::vector<ModuleSpec>>> async_result =
      std::async(std::launch::async,
                 [&] { return client.GetModulesInfo(file_spec, triple); });
  HandlePacket(
      server,
      "jModulesInfo:["
      R"({"file":"/foo/bar.so","triple":"i386-pc-linux"}])",
      R"([{"uuid":"404142434445464748494a4b4c4d4e4f50515253","triple":"i386-pc-linux",)"
      R"("file_path":"/foo/bar.so","file_offset":0,"file_size":1234}]])");

  auto result = async_result.get();
  ASSERT_TRUE(result.hasValue());
  ASSERT_EQ(1u, result->size());
  EXPECT_EQ("/foo/bar.so", result.getValue()[0].GetFileSpec().GetPath());
  EXPECT_EQ(triple, result.getValue()[0].GetArchitecture().GetTriple());
  EXPECT_EQ(UUID::fromData("@ABCDEFGHIJKLMNOPQRS", 20),
            result.getValue()[0].GetUUID());
  EXPECT_EQ(0u, result.getValue()[0].GetObjectOffset());
  EXPECT_EQ(1234u, result.getValue()[0].GetObjectSize());
}

TEST_F(GDBRemoteCommunicationClientTest, GetModulesInfoInvalidResponse) {
  llvm::Triple triple("i386-pc-linux");
  FileSpec file_spec("/foo/bar.so", FileSpec::Style::posix);

  const char *invalid_responses[] = {
      // no UUID
      R"([{"triple":"i386-pc-linux",)"
      R"("file_path":"/foo/bar.so","file_offset":0,"file_size":1234}]])",
      // invalid UUID
      R"([{"uuid":"XXXXXX","triple":"i386-pc-linux",)"
      R"("file_path":"/foo/bar.so","file_offset":0,"file_size":1234}]])",
      // no triple
      R"([{"uuid":"404142434445464748494a4b4c4d4e4f",)"
      R"("file_path":"/foo/bar.so","file_offset":0,"file_size":1234}]])",
      // no file_path
      R"([{"uuid":"404142434445464748494a4b4c4d4e4f","triple":"i386-pc-linux",)"
      R"("file_offset":0,"file_size":1234}]])",
      // no file_offset
      R"([{"uuid":"404142434445464748494a4b4c4d4e4f","triple":"i386-pc-linux",)"
      R"("file_path":"/foo/bar.so","file_size":1234}]])",
      // no file_size
      R"([{"uuid":"404142434445464748494a4b4c4d4e4f","triple":"i386-pc-linux",)"
      R"("file_path":"/foo/bar.so","file_offset":0}]])",
  };

  for (const char *response : invalid_responses) {
    std::future<llvm::Optional<std::vector<ModuleSpec>>> async_result =
        std::async(std::launch::async,
                   [&] { return client.GetModulesInfo(file_spec, triple); });
    HandlePacket(
        server,
        R"(jModulesInfo:[{"file":"/foo/bar.so","triple":"i386-pc-linux"}])",
        response);

    auto result = async_result.get();
    ASSERT_TRUE(result);
    ASSERT_EQ(0u, result->size()) << "response was: " << response;
  }
}

TEST_F(GDBRemoteCommunicationClientTest, TestPacketSpeedJSON) {
  std::thread server_thread([this] {
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
  auto object_sp = StructuredData::ParseJSON(std::string(ss.GetString()));
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
  std::future<Status> result = std::async(std::launch::async, [&] {
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

TEST_F(GDBRemoteCommunicationClientTest, GetMemoryRegionInfo) {
  const lldb::addr_t addr = 0xa000;
  MemoryRegionInfo region_info;
  std::future<Status> result = std::async(std::launch::async, [&] {
    return client.GetMemoryRegionInfo(addr, region_info);
  });

  HandlePacket(server,
      "qMemoryRegionInfo:a000",
      "start:a000;size:2000;permissions:rx;name:2f666f6f2f6261722e736f;");
  if (XMLDocument::XMLEnabled()) {
    // In case we have XML support, this will also do a "qXfer:memory-map".
    // Preceeded by a query for supported extensions. Pretend we don't support
    // that.
    HandlePacket(server, testing::StartsWith("qSupported:"), "");
  }
  EXPECT_TRUE(result.get().Success());
  EXPECT_EQ(addr, region_info.GetRange().GetRangeBase());
  EXPECT_EQ(0x2000u, region_info.GetRange().GetByteSize());
  EXPECT_EQ(MemoryRegionInfo::eYes, region_info.GetReadable());
  EXPECT_EQ(MemoryRegionInfo::eNo, region_info.GetWritable());
  EXPECT_EQ(MemoryRegionInfo::eYes, region_info.GetExecutable());
  EXPECT_EQ("/foo/bar.so", region_info.GetName().GetStringRef());
  EXPECT_EQ(MemoryRegionInfo::eDontKnow, region_info.GetMemoryTagged());

  result = std::async(std::launch::async, [&] {
    return client.GetMemoryRegionInfo(addr, region_info);
  });

  HandlePacket(server, "qMemoryRegionInfo:a000",
               "start:a000;size:2000;flags:;");
  EXPECT_TRUE(result.get().Success());
  EXPECT_EQ(MemoryRegionInfo::eNo, region_info.GetMemoryTagged());

  result = std::async(std::launch::async, [&] {
    return client.GetMemoryRegionInfo(addr, region_info);
  });

  HandlePacket(server, "qMemoryRegionInfo:a000",
               "start:a000;size:2000;flags: mt  zz mt  ;");
  EXPECT_TRUE(result.get().Success());
  EXPECT_EQ(MemoryRegionInfo::eYes, region_info.GetMemoryTagged());
}

TEST_F(GDBRemoteCommunicationClientTest, GetMemoryRegionInfoInvalidResponse) {
  const lldb::addr_t addr = 0x4000;
  MemoryRegionInfo region_info;
  std::future<Status> result = std::async(std::launch::async, [&] {
    return client.GetMemoryRegionInfo(addr, region_info);
  });

  HandlePacket(server, "qMemoryRegionInfo:4000", "start:4000;size:0000;");
  if (XMLDocument::XMLEnabled()) {
    // In case we have XML support, this will also do a "qXfer:memory-map".
    // Preceeded by a query for supported extensions. Pretend we don't support
    // that.
    HandlePacket(server, testing::StartsWith("qSupported:"), "");
  }
  EXPECT_FALSE(result.get().Success());
}

TEST_F(GDBRemoteCommunicationClientTest, SendTraceSupportedPacket) {
  TraceSupportedResponse trace_type;
  std::string error_message;
  auto callback = [&] {
    std::chrono::seconds timeout(10);
    if (llvm::Expected<TraceSupportedResponse> trace_type_or_err =
            client.SendTraceSupported(timeout)) {
      trace_type = *trace_type_or_err;
      error_message = "";
      return true;
    } else {
      trace_type = {};
      error_message = llvm::toString(trace_type_or_err.takeError());
      return false;
    }
  };

  // Success response
  {
    std::future<bool> result = std::async(std::launch::async, callback);

    HandlePacket(
        server, "jLLDBTraceSupported",
        R"({"name":"intel-pt","description":"Intel Processor Trace"}])");

    EXPECT_TRUE(result.get());
    ASSERT_STREQ(trace_type.name.c_str(), "intel-pt");
    ASSERT_STREQ(trace_type.description.c_str(), "Intel Processor Trace");
  }

  // Error response - wrong json
  {
    std::future<bool> result = std::async(std::launch::async, callback);

    HandlePacket(server, "jLLDBTraceSupported", R"({"type":"intel-pt"}])");

    EXPECT_FALSE(result.get());
    ASSERT_STREQ(error_message.c_str(), "missing value at TraceSupportedResponse.description");
  }

  // Error response
  {
    std::future<bool> result = std::async(std::launch::async, callback);

    HandlePacket(server, "jLLDBTraceSupported", "E23");

    EXPECT_FALSE(result.get());
  }

  // Error response with error message
  {
    std::future<bool> result = std::async(std::launch::async, callback);

    HandlePacket(server, "jLLDBTraceSupported",
                 "E23;50726F63657373206E6F742072756E6E696E672E");

    EXPECT_FALSE(result.get());
    ASSERT_STREQ(error_message.c_str(), "Process not running.");
  }
}

TEST_F(GDBRemoteCommunicationClientTest, GetQOffsets) {
  const auto &GetQOffsets = [&](llvm::StringRef response) {
    std::future<Optional<QOffsets>> result = std::async(
        std::launch::async, [&] { return client.GetQOffsets(); });

    HandlePacket(server, "qOffsets", response);
    return result.get();
  };
  EXPECT_EQ((QOffsets{false, {0x1234, 0x1234}}),
            GetQOffsets("Text=1234;Data=1234"));
  EXPECT_EQ((QOffsets{false, {0x1234, 0x1234, 0x1234}}),
            GetQOffsets("Text=1234;Data=1234;Bss=1234"));
  EXPECT_EQ((QOffsets{true, {0x1234}}), GetQOffsets("TextSeg=1234"));
  EXPECT_EQ((QOffsets{true, {0x1234, 0x2345}}),
            GetQOffsets("TextSeg=1234;DataSeg=2345"));

  EXPECT_EQ(llvm::None, GetQOffsets("E05"));
  EXPECT_EQ(llvm::None, GetQOffsets("Text=bogus"));
  EXPECT_EQ(llvm::None, GetQOffsets("Text=1234"));
  EXPECT_EQ(llvm::None, GetQOffsets("Text=1234;Data=1234;"));
  EXPECT_EQ(llvm::None, GetQOffsets("Text=1234;Data=1234;Bss=1234;"));
  EXPECT_EQ(llvm::None, GetQOffsets("TEXTSEG=1234"));
  EXPECT_EQ(llvm::None, GetQOffsets("TextSeg=0x1234"));
  EXPECT_EQ(llvm::None, GetQOffsets("TextSeg=12345678123456789"));
}

static void
check_qmemtags(TestClient &client, MockServer &server, size_t read_len,
               int32_t type, const char *packet, llvm::StringRef response,
               llvm::Optional<std::vector<uint8_t>> expected_tag_data) {
  const auto &ReadMemoryTags = [&]() {
    std::future<DataBufferSP> result = std::async(std::launch::async, [&] {
      return client.ReadMemoryTags(0xDEF0, read_len, type);
    });

    HandlePacket(server, packet, response);
    return result.get();
  };

  auto result = ReadMemoryTags();
  if (expected_tag_data) {
    ASSERT_TRUE(result);
    llvm::ArrayRef<uint8_t> expected(*expected_tag_data);
    llvm::ArrayRef<uint8_t> got = result->GetData();
    ASSERT_THAT(expected, testing::ContainerEq(got));
  } else {
    ASSERT_FALSE(result);
  }
}

TEST_F(GDBRemoteCommunicationClientTest, ReadMemoryTags) {
  // Zero length reads are valid
  check_qmemtags(client, server, 0, 1, "qMemTags:def0,0:1", "m",
                 std::vector<uint8_t>{});

  // Type can be negative. Put into the packet as the raw bytes
  // (as opposed to a literal -1)
  check_qmemtags(client, server, 0, -1, "qMemTags:def0,0:ffffffff", "m",
                 std::vector<uint8_t>{});
  check_qmemtags(client, server, 0, std::numeric_limits<int32_t>::min(),
                 "qMemTags:def0,0:80000000", "m", std::vector<uint8_t>{});
  check_qmemtags(client, server, 0, std::numeric_limits<int32_t>::max(),
                 "qMemTags:def0,0:7fffffff", "m", std::vector<uint8_t>{});

  // The client layer does not check the length of the received data.
  // All we need is the "m" and for the decode to use all of the chars
  check_qmemtags(client, server, 32, 2, "qMemTags:def0,20:2", "m09",
                 std::vector<uint8_t>{0x9});

  // Zero length response is fine as long as the "m" is present
  check_qmemtags(client, server, 0, 0x34, "qMemTags:def0,0:34", "m",
                 std::vector<uint8_t>{});

  // Normal responses
  check_qmemtags(client, server, 16, 1, "qMemTags:def0,10:1", "m66",
                 std::vector<uint8_t>{0x66});
  check_qmemtags(client, server, 32, 1, "qMemTags:def0,20:1", "m0102",
                 std::vector<uint8_t>{0x1, 0x2});

  // Empty response is an error
  check_qmemtags(client, server, 17, 1, "qMemTags:def0,11:1", "", llvm::None);
  // Usual error response
  check_qmemtags(client, server, 17, 1, "qMemTags:def0,11:1", "E01",
                 llvm::None);
  // Leading m missing
  check_qmemtags(client, server, 17, 1, "qMemTags:def0,11:1", "01", llvm::None);
  // Anything other than m is an error
  check_qmemtags(client, server, 17, 1, "qMemTags:def0,11:1", "z01",
                 llvm::None);
  // Decoding tag data doesn't use all the chars in the packet
  check_qmemtags(client, server, 32, 1, "qMemTags:def0,20:1", "m09zz",
                 llvm::None);
  // Data that is not hex bytes
  check_qmemtags(client, server, 32, 1, "qMemTags:def0,20:1", "mhello",
                 llvm::None);
  // Data is not a complete hex char
  check_qmemtags(client, server, 32, 1, "qMemTags:def0,20:1", "m9", llvm::None);
  // Data has a trailing hex char
  check_qmemtags(client, server, 32, 1, "qMemTags:def0,20:1", "m01020",
                 llvm::None);
}

static void check_Qmemtags(TestClient &client, MockServer &server,
                           lldb::addr_t addr, size_t len, int32_t type,
                           const std::vector<uint8_t> &tags, const char *packet,
                           llvm::StringRef response, bool should_succeed) {
  const auto &WriteMemoryTags = [&]() {
    std::future<Status> result = std::async(std::launch::async, [&] {
      return client.WriteMemoryTags(addr, len, type, tags);
    });

    HandlePacket(server, packet, response);
    return result.get();
  };

  auto result = WriteMemoryTags();
  if (should_succeed)
    ASSERT_TRUE(result.Success());
  else
    ASSERT_TRUE(result.Fail());
}

TEST_F(GDBRemoteCommunicationClientTest, WriteMemoryTags) {
  check_Qmemtags(client, server, 0xABCD, 0x20, 1,
                 std::vector<uint8_t>{0x12, 0x34}, "QMemTags:abcd,20:1:1234",
                 "OK", true);

  // The GDB layer doesn't care that the number of tags !=
  // the length of the write.
  check_Qmemtags(client, server, 0x4321, 0x20, 9, std::vector<uint8_t>{},
                 "QMemTags:4321,20:9:", "OK", true);

  check_Qmemtags(client, server, 0x8877, 0x123, 0x34,
                 std::vector<uint8_t>{0x55, 0x66, 0x77},
                 "QMemTags:8877,123:34:556677", "E01", false);

  // Type is a signed integer but is packed as its raw bytes,
  // instead of having a +/-.
  check_Qmemtags(client, server, 0x456789, 0, -1, std::vector<uint8_t>{0x99},
                 "QMemTags:456789,0:ffffffff:99", "E03", false);
  check_Qmemtags(client, server, 0x456789, 0,
                 std::numeric_limits<int32_t>::max(),
                 std::vector<uint8_t>{0x99}, "QMemTags:456789,0:7fffffff:99",
                 "E03", false);
  check_Qmemtags(client, server, 0x456789, 0,
                 std::numeric_limits<int32_t>::min(),
                 std::vector<uint8_t>{0x99}, "QMemTags:456789,0:80000000:99",
                 "E03", false);
}
