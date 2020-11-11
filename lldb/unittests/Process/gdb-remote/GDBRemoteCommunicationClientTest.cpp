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
#include "lldb/Utility/TraceOptions.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include <future>

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

TEST_F(GDBRemoteCommunicationClientTest, SendTraceSupportedTypePacket) {
  TraceTypeInfo trace_type;
  std::string error_message;
  auto callback = [&] {
    if (llvm::Expected<TraceTypeInfo> trace_type_or_err =
            client.SendGetSupportedTraceType()) {
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
        server, "jLLDBTraceSupportedType",
        R"({"name":"intel-pt","description":"Intel Processor Trace"}])");

    EXPECT_TRUE(result.get());
    ASSERT_STREQ(trace_type.name.c_str(), "intel-pt");
    ASSERT_STREQ(trace_type.description.c_str(), "Intel Processor Trace");
  }

  // Error response - wrong json
  {
    std::future<bool> result = std::async(std::launch::async, callback);

    HandlePacket(server, "jLLDBTraceSupportedType", R"({"type":"intel-pt"}])");

    EXPECT_FALSE(result.get());
    ASSERT_STREQ(error_message.c_str(), "missing value at (root).name");
  }

  // Error response
  {
    std::future<bool> result = std::async(std::launch::async, callback);

    HandlePacket(server, "jLLDBTraceSupportedType", "E23");

    EXPECT_FALSE(result.get());
  }

  // Error response with error message
  {
    std::future<bool> result = std::async(std::launch::async, callback);

    HandlePacket(server, "jLLDBTraceSupportedType",
                 "E23;50726F63657373206E6F742072756E6E696E672E");

    EXPECT_FALSE(result.get());
    ASSERT_STREQ(error_message.c_str(), "Process not running.");
  }
}

TEST_F(GDBRemoteCommunicationClientTest, SendStartTracePacket) {
  TraceOptions options;
  Status error;

  options.setType(lldb::TraceType::eTraceTypeProcessorTrace);
  options.setMetaDataBufferSize(8192);
  options.setTraceBufferSize(8192);
  options.setThreadID(0x23);

  StructuredData::DictionarySP custom_params =
      std::make_shared<StructuredData::Dictionary>();
  custom_params->AddStringItem("tracetech", "intel-pt");
  custom_params->AddIntegerItem("psb", 0x01);

  options.setTraceParams(custom_params);

  std::future<lldb::user_id_t> result = std::async(std::launch::async, [&] {
    return client.SendStartTracePacket(options, error);
  });

  // Since the line is exceeding 80 characters.
  std::string expected_packet1 =
      R"(jTraceStart:{"buffersize":8192,"metabuffersize":8192,"params":)";
  std::string expected_packet2 =
      R"({"psb":1,"tracetech":"intel-pt"},"threadid":35,"type":1})";
  HandlePacket(server, (expected_packet1 + expected_packet2), "1");
  ASSERT_TRUE(error.Success());
  ASSERT_EQ(result.get(), 1u);

  error.Clear();
  result = std::async(std::launch::async, [&] {
    return client.SendStartTracePacket(options, error);
  });

  HandlePacket(server, (expected_packet1 + expected_packet2), "E23");
  ASSERT_EQ(result.get(), LLDB_INVALID_UID);
  ASSERT_FALSE(error.Success());
}

TEST_F(GDBRemoteCommunicationClientTest, SendStopTracePacket) {
  lldb::tid_t thread_id = 0x23;
  lldb::user_id_t trace_id = 3;

  std::future<Status> result = std::async(std::launch::async, [&] {
    return client.SendStopTracePacket(trace_id, thread_id);
  });

  const char *expected_packet = R"(jTraceStop:{"threadid":35,"traceid":3})";
  HandlePacket(server, expected_packet, "OK");
  ASSERT_TRUE(result.get().Success());

  result = std::async(std::launch::async, [&] {
    return client.SendStopTracePacket(trace_id, thread_id);
  });

  HandlePacket(server, expected_packet, "E23");
  ASSERT_FALSE(result.get().Success());
}

TEST_F(GDBRemoteCommunicationClientTest, SendGetDataPacket) {
  lldb::tid_t thread_id = 0x23;
  lldb::user_id_t trace_id = 3;

  uint8_t buf[32] = {};
  llvm::MutableArrayRef<uint8_t> buffer(buf, 32);
  size_t offset = 0;

  std::future<Status> result = std::async(std::launch::async, [&] {
    return client.SendGetDataPacket(trace_id, thread_id, buffer, offset);
  });

  std::string expected_packet1 =
      R"(jTraceBufferRead:{"buffersize":32,"offset":0,"threadid":35,)";
  std::string expected_packet2 = R"("traceid":3})";
  HandlePacket(server, expected_packet1+expected_packet2, "123456");
  ASSERT_TRUE(result.get().Success());
  ASSERT_EQ(buffer.size(), 3u);
  ASSERT_EQ(buf[0], 0x12);
  ASSERT_EQ(buf[1], 0x34);
  ASSERT_EQ(buf[2], 0x56);

  llvm::MutableArrayRef<uint8_t> buffer2(buf, 32);
  result = std::async(std::launch::async, [&] {
    return client.SendGetDataPacket(trace_id, thread_id, buffer2, offset);
  });

  HandlePacket(server, expected_packet1+expected_packet2, "E23");
  ASSERT_FALSE(result.get().Success());
  ASSERT_EQ(buffer2.size(), 0u);
}

TEST_F(GDBRemoteCommunicationClientTest, SendGetMetaDataPacket) {
  lldb::tid_t thread_id = 0x23;
  lldb::user_id_t trace_id = 3;

  uint8_t buf[32] = {};
  llvm::MutableArrayRef<uint8_t> buffer(buf, 32);
  size_t offset = 0;

  std::future<Status> result = std::async(std::launch::async, [&] {
    return client.SendGetMetaDataPacket(trace_id, thread_id, buffer, offset);
  });

  std::string expected_packet1 =
      R"(jTraceMetaRead:{"buffersize":32,"offset":0,"threadid":35,)";
  std::string expected_packet2 = R"("traceid":3})";
  HandlePacket(server, expected_packet1+expected_packet2, "123456");
  ASSERT_TRUE(result.get().Success());
  ASSERT_EQ(buffer.size(), 3u);
  ASSERT_EQ(buf[0], 0x12);
  ASSERT_EQ(buf[1], 0x34);
  ASSERT_EQ(buf[2], 0x56);

  llvm::MutableArrayRef<uint8_t> buffer2(buf, 32);
  result = std::async(std::launch::async, [&] {
    return client.SendGetMetaDataPacket(trace_id, thread_id, buffer2, offset);
  });

  HandlePacket(server, expected_packet1+expected_packet2, "E23");
  ASSERT_FALSE(result.get().Success());
  ASSERT_EQ(buffer2.size(), 0u);
}

TEST_F(GDBRemoteCommunicationClientTest, SendGetTraceConfigPacket) {
  lldb::tid_t thread_id = 0x23;
  lldb::user_id_t trace_id = 3;
  TraceOptions options;
  options.setThreadID(thread_id);

  std::future<Status> result = std::async(std::launch::async, [&] {
    return client.SendGetTraceConfigPacket(trace_id, options);
  });

  const char *expected_packet =
      R"(jTraceConfigRead:{"threadid":35,"traceid":3})";
  std::string response1 =
      R"({"buffersize":8192,"params":{"psb":1,"tracetech":"intel-pt"})";
  std::string response2 = R"(],"metabuffersize":8192,"threadid":35,"type":1}])";
  HandlePacket(server, expected_packet, response1+response2);
  ASSERT_TRUE(result.get().Success());
  ASSERT_EQ(options.getTraceBufferSize(), 8192u);
  ASSERT_EQ(options.getMetaDataBufferSize(), 8192u);
  ASSERT_EQ(options.getType(), 1);

  auto custom_params = options.getTraceParams();

  uint64_t psb_value;
  llvm::StringRef trace_tech_value;

  ASSERT_TRUE(custom_params);
  ASSERT_EQ(custom_params->GetType(), eStructuredDataTypeDictionary);
  ASSERT_TRUE(custom_params->GetValueForKeyAsInteger("psb", psb_value));
  ASSERT_EQ(psb_value, 1u);
  ASSERT_TRUE(
      custom_params->GetValueForKeyAsString("tracetech", trace_tech_value));
  ASSERT_STREQ(trace_tech_value.data(), "intel-pt");

  // Checking error response.
  std::future<Status> result2 = std::async(std::launch::async, [&] {
    return client.SendGetTraceConfigPacket(trace_id, options);
  });

  HandlePacket(server, expected_packet, "E23");
  ASSERT_FALSE(result2.get().Success());

  // Wrong JSON as response.
  std::future<Status> result3 = std::async(std::launch::async, [&] {
    return client.SendGetTraceConfigPacket(trace_id, options);
  });

  std::string incorrect_json1 =
      R"("buffersize" : 8192,"params" : {"psb" : 1,"tracetech" : "intel-pt"})";
  std::string incorrect_json2 =
      R"(],"metabuffersize" : 8192,"threadid" : 35,"type" : 1}])";
  HandlePacket(server, expected_packet, incorrect_json1+incorrect_json2);
  ASSERT_FALSE(result3.get().Success());

  // Wrong JSON as custom_params.
  std::future<Status> result4 = std::async(std::launch::async, [&] {
    return client.SendGetTraceConfigPacket(trace_id, options);
  });

  std::string incorrect_custom_params1 =
      R"({"buffersize" : 8192,"params" : "psb" : 1,"tracetech" : "intel-pt"})";
  std::string incorrect_custom_params2 =
      R"(],"metabuffersize" : 8192,"threadid" : 35,"type" : 1}])";
  HandlePacket(server, expected_packet, incorrect_custom_params1+
      incorrect_custom_params2);
  ASSERT_FALSE(result4.get().Success());
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
