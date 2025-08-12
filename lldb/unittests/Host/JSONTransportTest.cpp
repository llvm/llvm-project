//===----------------------------------------------------------------------===//
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
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <chrono>
#include <cstddef>
#include <future>
#include <memory>
#include <string>

using namespace llvm;
using namespace lldb_private;

namespace {

struct JSONTestType {
  std::string str;
};

json::Value toJSON(const JSONTestType &T) {
  return json::Object{{"str", T.str}};
}

bool fromJSON(const json::Value &V, JSONTestType &T, json::Path P) {
  json::ObjectMapper O(V, P);
  return O && O.map("str", T.str);
}

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
  Expected<P>
  RunOnce(std::chrono::milliseconds timeout = std::chrono::seconds(1)) {
    std::promise<Expected<P>> promised_message;
    std::future<Expected<P>> future_message = promised_message.get_future();
    RunUntil<P>(
        [&promised_message](Expected<P> message) mutable -> bool {
          promised_message.set_value(std::move(message));
          return /*keep_going*/ false;
        },
        timeout);
    return future_message.get();
  }

  /// RunUntil runs the event loop until the callback returns `false` or a
  /// timeout has occurred.
  template <typename P>
  void RunUntil(std::function<bool(Expected<P>)> callback,
                std::chrono::milliseconds timeout = std::chrono::seconds(1)) {
    auto handle = transport->RegisterReadObject<P>(
        loop, [&callback](MainLoopBase &loop, Expected<P> message) mutable {
          bool keep_going = callback(std::move(message));
          if (!keep_going)
            loop.RequestTermination();
        });
    loop.AddCallback(
        [&callback](MainLoopBase &loop) mutable {
          loop.RequestTermination();
          callback(createStringError("timeout"));
        },
        timeout);
    EXPECT_THAT_EXPECTED(handle, Succeeded());
    EXPECT_THAT_ERROR(loop.Run().takeError(), Succeeded());
  }

  template <typename... Ts> llvm::Expected<size_t> Write(Ts... args) {
    std::string message;
    for (const auto &arg : {args...})
      message += Encode(arg);
    return input.Write(message.data(), message.size());
  }

  virtual std::string Encode(const json::Value &) = 0;
};

class HTTPDelimitedJSONTransportTest
    : public JSONTransportTest<HTTPDelimitedJSONTransport> {
public:
  using JSONTransportTest::JSONTransportTest;

  std::string Encode(const json::Value &V) override {
    std::string msg;
    raw_string_ostream OS(msg);
    OS << formatv("{0}", V);
    return formatv("Content-Length: {0}\r\nContent-type: "
                   "text/json\r\n\r\n{1}",
                   msg.size(), msg)
        .str();
  }
};

class JSONRPCTransportTest : public JSONTransportTest<JSONRPCTransport> {
public:
  using JSONTransportTest::JSONTransportTest;

  std::string Encode(const json::Value &V) override {
    std::string msg;
    raw_string_ostream OS(msg);
    OS << formatv("{0}\n", V);
    return msg;
  }
};

} // namespace

TEST_F(HTTPDelimitedJSONTransportTest, MalformedRequests) {
  std::string malformed_header =
      "COnTent-LenGth: -1\r\nContent-Type: text/json\r\n\r\nnotjosn";
  ASSERT_THAT_EXPECTED(
      input.Write(malformed_header.data(), malformed_header.size()),
      Succeeded());
  ASSERT_THAT_EXPECTED(RunOnce<JSONTestType>(),
                       FailedWithMessage("invalid content length: -1"));
}

TEST_F(HTTPDelimitedJSONTransportTest, Read) {
  ASSERT_THAT_EXPECTED(Write(JSONTestType{"foo"}), Succeeded());
  ASSERT_THAT_EXPECTED(RunOnce<JSONTestType>(),
                       HasValue(testing::FieldsAre(/*str=*/"foo")));
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadMultipleMessagesInSingleWrite) {
  ASSERT_THAT_EXPECTED(Write(JSONTestType{"one"}, JSONTestType{"two"}),
                       Succeeded());
  unsigned count = 0;
  RunUntil<JSONTestType>([&](Expected<JSONTestType> message) -> bool {
    if (count == 0) {
      EXPECT_THAT_EXPECTED(message,
                           HasValue(testing::FieldsAre(/*str=*/"one")));
    } else if (count == 1) {
      EXPECT_THAT_EXPECTED(message,
                           HasValue(testing::FieldsAre(/*str=*/"two")));
    }

    count++;
    return count < 2;
  });
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadAcrossMultipleChunks) {
  std::string long_str = std::string(2048, 'x');
  ASSERT_THAT_EXPECTED(Write(JSONTestType{long_str}), Succeeded());
  ASSERT_THAT_EXPECTED(RunOnce<JSONTestType>(),
                       HasValue(testing::FieldsAre(/*str=*/long_str)));
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadPartialMessage) {
  std::string message = Encode(JSONTestType{"foo"});
  std::string part1 = message.substr(0, 28);
  std::string part2 = message.substr(28);

  ASSERT_THAT_EXPECTED(input.Write(part1.data(), part1.size()), Succeeded());

  ASSERT_THAT_EXPECTED(
      RunOnce<JSONTestType>(/*timeout=*/std::chrono::milliseconds(10)),
      FailedWithMessage("timeout"));

  ASSERT_THAT_EXPECTED(input.Write(part2.data(), part2.size()), Succeeded());

  ASSERT_THAT_EXPECTED(RunOnce<JSONTestType>(),
                       HasValue(testing::FieldsAre(/*str=*/"foo")));
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadWithZeroByteWrites) {
  std::string message = Encode(JSONTestType{"foo"});
  std::string part1 = message.substr(0, 28);
  std::string part2 = message.substr(28);

  ASSERT_THAT_EXPECTED(input.Write(part1.data(), part1.size()), Succeeded());
  ASSERT_THAT_EXPECTED(
      RunOnce<JSONTestType>(/*timeout=*/std::chrono::milliseconds(10)),
      FailedWithMessage("timeout"));

  ASSERT_THAT_EXPECTED(input.Write(part1.data(), 0),
                       Succeeded()); // zero-byte write.

  ASSERT_THAT_EXPECTED(
      RunOnce<JSONTestType>(/*timeout=*/std::chrono::milliseconds(10)),
      FailedWithMessage("timeout"));

  ASSERT_THAT_EXPECTED(input.Write(part2.data(), part2.size()), Succeeded());

  ASSERT_THAT_EXPECTED(RunOnce<JSONTestType>(),
                       HasValue(testing::FieldsAre(/*str=*/"foo")));
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadWithEOF) {
  input.CloseWriteFileDescriptor();
  ASSERT_THAT_EXPECTED(RunOnce<JSONTestType>(), Failed<TransportEOFError>());
}

TEST_F(HTTPDelimitedJSONTransportTest, ReaderWithUnhandledData) {
  std::string json = R"json({"str": "foo"})json";
  std::string message =
      formatv("Content-Length: {0}\r\nContent-type: text/json\r\n\r\n{1}",
              json.size(), json)
          .str();
  // Write an incomplete message and close the handle.
  ASSERT_THAT_EXPECTED(input.Write(message.data(), message.size() - 1),
                       Succeeded());
  input.CloseWriteFileDescriptor();
  ASSERT_THAT_EXPECTED(RunOnce<JSONTestType>(),
                       Failed<TransportUnhandledContentsError>());
}

TEST_F(HTTPDelimitedJSONTransportTest, NoDataTimeout) {
  ASSERT_THAT_EXPECTED(
      RunOnce<JSONTestType>(/*timeout=*/std::chrono::milliseconds(10)),
      FailedWithMessage("timeout"));
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
  ASSERT_THAT_EXPECTED(RunOnce<JSONTestType>(), llvm::Failed());
}

TEST_F(JSONRPCTransportTest, Read) {
  ASSERT_THAT_EXPECTED(Write(JSONTestType{"foo"}), Succeeded());
  ASSERT_THAT_EXPECTED(RunOnce<JSONTestType>(),
                       HasValue(testing::FieldsAre(/*str=*/"foo")));
}

TEST_F(JSONRPCTransportTest, ReadAcrossMultipleChunks) {
  std::string long_str = std::string(2048, 'x');
  std::string message = Encode(JSONTestType{long_str});
  ASSERT_THAT_EXPECTED(input.Write(message.data(), message.size()),
                       Succeeded());
  ASSERT_THAT_EXPECTED(RunOnce<JSONTestType>(),
                       HasValue(testing::FieldsAre(/*str=*/long_str)));
}

TEST_F(JSONRPCTransportTest, ReadPartialMessage) {
  std::string message = R"({"str": "foo"})"
                        "\n";
  std::string part1 = message.substr(0, 7);
  std::string part2 = message.substr(7);

  ASSERT_THAT_EXPECTED(input.Write(part1.data(), part1.size()), Succeeded());

  ASSERT_THAT_EXPECTED(
      RunOnce<JSONTestType>(/*timeout=*/std::chrono::milliseconds(10)),
      FailedWithMessage("timeout"));

  ASSERT_THAT_EXPECTED(input.Write(part2.data(), part2.size()), Succeeded());

  ASSERT_THAT_EXPECTED(RunOnce<JSONTestType>(),
                       HasValue(testing::FieldsAre(/*str=*/"foo")));
}

TEST_F(JSONRPCTransportTest, ReadWithEOF) {
  input.CloseWriteFileDescriptor();
  ASSERT_THAT_EXPECTED(RunOnce<JSONTestType>(), Failed<TransportEOFError>());
}

TEST_F(JSONRPCTransportTest, ReaderWithUnhandledData) {
  std::string message = R"json({"str": "foo"})json"
                        "\n";
  // Write an incomplete message and close the handle.
  ASSERT_THAT_EXPECTED(input.Write(message.data(), message.size() - 1),
                       Succeeded());
  input.CloseWriteFileDescriptor();
  ASSERT_THAT_EXPECTED(RunOnce<JSONTestType>(),
                       Failed<TransportUnhandledContentsError>());
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

TEST_F(JSONRPCTransportTest, NoDataTimeout) {
  ASSERT_THAT_EXPECTED(
      RunOnce<JSONTestType>(/*timeout=*/std::chrono::milliseconds(10)),
      FailedWithMessage("timeout"));
}
