//===-- JSONTransportTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/JSONTransport.h"
#include "TestingSupport/Host/PipeTestUtilities.h"
#include "lldb/Host/File.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/MainLoopBase.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <chrono>
#include <future>
#include <thread>

using namespace llvm;
using namespace lldb_private;

namespace {
template <typename T> class JSONTransportTest : public PipePairTest {
protected:
  std::unique_ptr<JSONTransport> transport;
  MainLoop loop;

  void SetUp() override {
    PipePairTest::SetUp();
    transport = std::make_unique<T>(
        std::make_shared<NativeFile>(input.GetReadFileDescriptor(),
                                     File::eOpenOptionReadOnly,
                                     NativeFile::Unowned),
        std::make_shared<NativeFile>(output.GetWriteFileDescriptor(),
                                     File::eOpenOptionWriteOnly,
                                     NativeFile::Unowned));
  }

  template <typename P>
  void
  RunOnce(std::function<void(llvm::Expected<P>)> callback,
          std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
    auto handle = transport->RegisterReadObject<P>(
        loop, [&](MainLoopBase &loop, llvm::Expected<P> message) {
          callback(std::move(message));
          loop.RequestTermination();
        });
    loop.AddCallback(
        [&](MainLoopBase &loop) {
          loop.RequestTermination();
          FAIL() << "timeout waiting for read callback";
        },
        timeout);
    ASSERT_THAT_EXPECTED(handle, Succeeded());
    ASSERT_THAT_ERROR(loop.Run().takeError(), Succeeded());
  }
};

class HTTPDelimitedJSONTransportTest
    : public JSONTransportTest<HTTPDelimitedJSONTransport> {
public:
  using JSONTransportTest::JSONTransportTest;
};

class JSONRPCTransportTest : public JSONTransportTest<JSONRPCTransport> {
public:
  using JSONTransportTest::JSONTransportTest;
};

struct JSONTestType {
  std::string str;
};

llvm::json::Value toJSON(const JSONTestType &T) {
  return llvm::json::Object{{"str", T.str}};
}

bool fromJSON(const llvm::json::Value &V, JSONTestType &T, llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.map("str", T.str);
}
} // namespace

TEST_F(HTTPDelimitedJSONTransportTest, MalformedRequests) {
  std::string malformed_header =
      "COnTent-LenGth: -1\r\nContent-Type: text/json\r\n\r\nnotjosn";
  ASSERT_THAT_EXPECTED(
      input.Write(malformed_header.data(), malformed_header.size()),
      Succeeded());
  RunOnce<JSONTestType>([&](llvm::Expected<JSONTestType> message) {
    ASSERT_THAT_EXPECTED(message,
                         FailedWithMessage("invalid content length: -1"));
  });
}

TEST_F(HTTPDelimitedJSONTransportTest, Read) {
  std::string json = R"json({"str": "foo"})json";
  std::string message =
      formatv("Content-Length: {0}\r\nContent-type: text/json\r\n\r\n{1}",
              json.size(), json)
          .str();
  ASSERT_THAT_EXPECTED(input.Write(message.data(), message.size()),
                       Succeeded());
  RunOnce<JSONTestType>([&](llvm::Expected<JSONTestType> message) {
    ASSERT_THAT_EXPECTED(message, HasValue(testing::FieldsAre(/*str=*/"foo")));
  });
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadAcrossMultipleChunks) {
  std::string long_str = std::string(2048, 'x');
  std::string json = formatv(R"({"str": "{0}"})", long_str).str();
  std::string message =
      formatv("Content-Length: {0}\r\n\r\n{1}", json.size(), json).str();
  ASSERT_THAT_EXPECTED(input.Write(message.data(), message.size()),
                       Succeeded());
  RunOnce<JSONTestType>([&](llvm::Expected<JSONTestType> message) {
    ASSERT_THAT_EXPECTED(message,
                         HasValue(testing::FieldsAre(/*str=*/long_str)));
  });
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadPartialMessage) {
  std::future<void> background_task = std::async(std::launch::async, [&]() {
    std::string json = R"({"str": "foo"})";
    std::string message =
        formatv("Content-Length: {0}\r\n\r\n{1}", json.size(), json).str();
    std::string part1 = message.substr(0, 28);
    std::string part2 = message.substr(28);

    ASSERT_THAT_EXPECTED(input.Write(part1.data(), part1.size()), Succeeded());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ASSERT_THAT_EXPECTED(input.Write(part2.data(), part2.size()), Succeeded());
  });

  RunOnce<JSONTestType>([&](llvm::Expected<JSONTestType> message) {
    ASSERT_THAT_EXPECTED(message, HasValue(testing::FieldsAre(/*str=*/"foo")));
  });
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadWithEOF) {
  input.CloseWriteFileDescriptor();
  RunOnce<JSONTestType>([&](llvm::Expected<JSONTestType> message) {
    ASSERT_THAT_EXPECTED(message, Failed<TransportEOFError>());
  });
}

TEST_F(HTTPDelimitedJSONTransportTest, InvalidTransport) {
  transport = std::make_unique<HTTPDelimitedJSONTransport>(nullptr, nullptr);
  auto handle = transport->RegisterReadObject<JSONTestType>(
      loop, [&](MainLoopBase &, llvm::Expected<JSONTestType>) {});
  ASSERT_THAT_EXPECTED(handle, FailedWithMessage("IO object is not valid."));
}

TEST_F(HTTPDelimitedJSONTransportTest, Write) {
  ASSERT_THAT_ERROR(transport->Write(JSONTestType{"foo"}), Succeeded());
  output.CloseWriteFileDescriptor();
  char buf[1024];
  Expected<size_t> bytes_read =
      output.Read(buf, sizeof(buf), std::chrono::milliseconds(1));
  ASSERT_THAT_EXPECTED(bytes_read, Succeeded());
  ASSERT_EQ(StringRef(buf, *bytes_read), StringRef("Content-Length: 13\r\n\r\n"
                                                   R"json({"str":"foo"})json"));
}

TEST_F(JSONRPCTransportTest, MalformedRequests) {
  std::string malformed_header = "notjson\n";
  ASSERT_THAT_EXPECTED(
      input.Write(malformed_header.data(), malformed_header.size()),
      Succeeded());
  RunOnce<JSONTestType>(
      [&](auto message) { ASSERT_THAT_EXPECTED(message, llvm::Failed()); });
}

TEST_F(JSONRPCTransportTest, Read) {
  std::string json = R"json({"str": "foo"})json";
  std::string message = formatv("{0}\n", json).str();
  ASSERT_THAT_EXPECTED(input.Write(message.data(), message.size()),
                       Succeeded());
  RunOnce<JSONTestType>([&](auto message) {
    ASSERT_THAT_EXPECTED(message, HasValue(testing::FieldsAre(/*str=*/"foo")));
  });
}

TEST_F(JSONRPCTransportTest, ReadAcrossMultipleChunks) {
  std::string long_str = std::string(2048, 'x');
  std::string json = formatv(R"({"str": "{0}"})", long_str).str();
  std::string message = formatv("{0}\n", json).str();
  ASSERT_THAT_EXPECTED(input.Write(message.data(), message.size()),
                       Succeeded());
  RunOnce<JSONTestType>([&](llvm::Expected<JSONTestType> message) {
    ASSERT_THAT_EXPECTED(message,
                         HasValue(testing::FieldsAre(/*str=*/long_str)));
  });
}

TEST_F(JSONRPCTransportTest, ReadPartialMessage) {
  std::future<void> background_task = std::async(std::launch::async, [&]() {
    std::string message = R"({"str": "foo"})"
                          "\n";
    std::string part1 = message.substr(0, 7);
    std::string part2 = message.substr(7);

    ASSERT_THAT_EXPECTED(input.Write(part1.data(), part1.size()), Succeeded());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ASSERT_THAT_EXPECTED(input.Write(part2.data(), part2.size()), Succeeded());
  });

  RunOnce<JSONTestType>([&](llvm::Expected<JSONTestType> message) {
    ASSERT_THAT_EXPECTED(message, HasValue(testing::FieldsAre(/*str=*/"foo")));
  });
}

TEST_F(JSONRPCTransportTest, ReadWithEOF) {
  input.CloseWriteFileDescriptor();
  RunOnce<JSONTestType>([&](llvm::Expected<JSONTestType> message) {
    ASSERT_THAT_EXPECTED(message, Failed<TransportEOFError>());
  });
}

TEST_F(JSONRPCTransportTest, Write) {
  ASSERT_THAT_ERROR(transport->Write(JSONTestType{"foo"}), Succeeded());
  output.CloseWriteFileDescriptor();
  char buf[1024];
  Expected<size_t> bytes_read =
      output.Read(buf, sizeof(buf), std::chrono::milliseconds(1));
  ASSERT_THAT_EXPECTED(bytes_read, Succeeded());
  ASSERT_EQ(StringRef(buf, *bytes_read), StringRef(R"json({"str":"foo"})json"
                                                   "\n"));
}

TEST_F(JSONRPCTransportTest, InvalidTransport) {
  transport = std::make_unique<JSONRPCTransport>(nullptr, nullptr);
  auto handle = transport->RegisterReadObject<JSONTestType>(
      loop, [&](MainLoopBase &, llvm::Expected<JSONTestType>) {});
  ASSERT_THAT_EXPECTED(handle, FailedWithMessage("IO object is not valid."));
}
