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
#include "lldb/Utility/Log.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <chrono>
#include <cstddef>
#include <memory>
#include <string>

using namespace llvm;
using namespace lldb_private;

namespace {

namespace test_protocol {

struct Req {
  std::string name;
};
json::Value toJSON(const Req &T) { return json::Object{{"req", T.name}}; }
bool fromJSON(const json::Value &V, Req &T, json::Path P) {
  json::ObjectMapper O(V, P);
  return O && O.map("req", T.name);
}
bool operator==(const Req &a, const Req &b) { return a.name == b.name; }
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Req &V) {
  OS << toJSON(V);
  return OS;
}
void PrintTo(const Req &message, std::ostream *os) {
  std::string O;
  llvm::raw_string_ostream OS(O);
  OS << message;
  *os << O;
}

struct Resp {
  std::string name;
};
json::Value toJSON(const Resp &T) { return json::Object{{"resp", T.name}}; }
bool fromJSON(const json::Value &V, Resp &T, json::Path P) {
  json::ObjectMapper O(V, P);
  return O && O.map("resp", T.name);
}
bool operator==(const Resp &a, const Resp &b) { return a.name == b.name; }
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Resp &V) {
  OS << toJSON(V);
  return OS;
}
void PrintTo(const Resp &message, std::ostream *os) {
  std::string O;
  llvm::raw_string_ostream OS(O);
  OS << message;
  *os << O;
}

struct Evt {
  std::string name;
};
json::Value toJSON(const Evt &T) { return json::Object{{"evt", T.name}}; }
bool fromJSON(const json::Value &V, Evt &T, json::Path P) {
  json::ObjectMapper O(V, P);
  return O && O.map("evt", T.name);
}
bool operator==(const Evt &a, const Evt &b) { return a.name == b.name; }
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Evt &V) {
  OS << toJSON(V);
  return OS;
}
void PrintTo(const Evt &message, std::ostream *os) {
  std::string O;
  llvm::raw_string_ostream OS(O);
  OS << message;
  *os << O;
}

using Message = std::variant<Req, Resp, Evt>;
json::Value toJSON(const Message &T) {
  if (const Req *req = std::get_if<Req>(&T))
    return toJSON(*req);
  if (const Resp *resp = std::get_if<Resp>(&T))
    return toJSON(*resp);
  if (const Evt *evt = std::get_if<Evt>(&T))
    return toJSON(*evt);
  llvm_unreachable("unknown message type");
}
bool fromJSON(const json::Value &V, Message &T, json::Path P) {
  const json::Object *O = V.getAsObject();
  if (!O) {
    P.report("expected object");
    return false;
  }
  if (O->get("req")) {
    Req R;
    if (!fromJSON(V, R, P))
      return false;

    T = std::move(R);
    return true;
  }
  if (O->get("resp")) {
    Resp R;
    if (!fromJSON(V, R, P))
      return false;

    T = std::move(R);
    return true;
  }
  if (O->get("evt")) {
    Evt E;
    if (!fromJSON(V, E, P))
      return false;

    T = std::move(E);
    return true;
  }
  P.report("unknown message type");
  return false;
}

} // namespace test_protocol

template <typename T, typename Req, typename Resp, typename Evt>
class JSONTransportTest : public PipePairTest {

protected:
  std::unique_ptr<T> transport;
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

  class MessageCollector final
      : public Transport<Req, Resp, Evt>::MessageHandler {
  public:
    MessageCollector(llvm::Error *err = nullptr) : err(err) {
      if (err)
        consumeError(std::move(*err));
    }
    std::vector<typename T::Message> messages;
    llvm::Error *err;
    void OnEvent(const Evt &V) override { messages.emplace_back(V); }
    void OnRequest(const Req &V) override { messages.emplace_back(V); }
    void OnResponse(const Resp &V) override { messages.emplace_back(V); }
    void OnError(MainLoopBase &loop, llvm::Error error) override {
      loop.RequestTermination();
      if (err)
        *err = std::move(error);
      else
        FAIL() << "Error while reading from transport: "
               << llvm::toString(std::move(error));
    }
    void OnEOF() override { /* no-op */ }
  };

  Expected<std::vector<typename T::Message>>
  Run(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
    return Run(nullptr, timeout);
  }

  /// Run the transport MainLoop and return any messages received.
  Expected<std::vector<typename T::Message>>
  Run(llvm::Error *err,
      std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
    MessageCollector collector(err);
    loop.AddCallback([](MainLoopBase &loop) { loop.RequestTermination(); },
                     timeout);
    auto handle = transport->RegisterMessageHandler(loop, collector);
    if (!handle)
      return handle.takeError();

    if (Status status = loop.Run(); status.Fail())
      return status.takeError();

    return std::move(collector.messages);
  }

  template <typename... Ts> void Write(Ts... args) {
    std::string message;
    for (const auto &arg : {args...})
      message += Encode(arg);
    EXPECT_THAT_EXPECTED(input.Write(message.data(), message.size()),
                         Succeeded());
  }

  template <typename... Ts> void WriteAndCloseInput(Ts... args) {
    Write<Ts...>(std::forward<Ts>(args)...);
    input.CloseWriteFileDescriptor();
  }

  virtual std::string Encode(const json::Value &) = 0;
};

class TestHTTPDelimitedJSONTransport final
    : public HTTPDelimitedJSONTransport<test_protocol::Req, test_protocol::Resp,
                                        test_protocol::Evt> {
public:
  using HTTPDelimitedJSONTransport::HTTPDelimitedJSONTransport;

  void Log(llvm::StringRef message) override {
    log_messages.emplace_back(message);
  }

  std::vector<std::string> log_messages;
};

class HTTPDelimitedJSONTransportTest
    : public JSONTransportTest<TestHTTPDelimitedJSONTransport,
                               test_protocol::Req, test_protocol::Resp,
                               test_protocol::Evt> {
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

class TestJSONRPCTransport final
    : public JSONRPCTransport<test_protocol::Req, test_protocol::Resp,
                              test_protocol::Evt> {
public:
  using JSONRPCTransport::JSONRPCTransport;

  void Log(llvm::StringRef message) override {
    log_messages.emplace_back(message);
  }

  std::vector<std::string> log_messages;
};

class JSONRPCTransportTest
    : public JSONTransportTest<TestJSONRPCTransport, test_protocol::Req,
                               test_protocol::Resp, test_protocol::Evt> {
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

// Failing on Windows, see https://github.com/llvm/llvm-project/issues/153446.
#ifndef _WIN32
using namespace test_protocol;

TEST_F(HTTPDelimitedJSONTransportTest, MalformedRequests) {
  std::string malformed_header =
      "COnTent-LenGth: -1\r\nContent-Type: text/json\r\n\r\nnotjosn";
  ASSERT_THAT_EXPECTED(
      input.Write(malformed_header.data(), malformed_header.size()),
      Succeeded());
  llvm::Error err = llvm::Error::success();
  ASSERT_THAT_EXPECTED(Run(&err), Succeeded());
  ASSERT_THAT_ERROR(std::move(err),
                    FailedWithMessage("invalid content length: -1"));
}

TEST_F(HTTPDelimitedJSONTransportTest, Read) {
  WriteAndCloseInput(Req{"foo"});
  ASSERT_THAT_EXPECTED(Run(), HasValue(testing::ElementsAre(Req{"foo"})));
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadMultipleMessagesInSingleWrite) {
  WriteAndCloseInput(Message{Req{"one"}}, Message{Resp{"two"}},
                     Message{Evt{"three"}});
  EXPECT_THAT_EXPECTED(Run(), HasValue(testing::ElementsAre(
                                  Req{"one"}, Resp{"two"}, Evt{"three"})));
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadAcrossMultipleChunks) {
  std::string long_str = std::string(2048, 'x');
  WriteAndCloseInput(Req{long_str});
  ASSERT_THAT_EXPECTED(Run(), HasValue(testing::ElementsAre(Req{long_str})));
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadPartialMessage) {
  std::string message = Encode(Req{"foo"});
  auto split_at = message.size() / 2;
  std::string part1 = message.substr(0, split_at);
  std::string part2 = message.substr(split_at);

  ASSERT_THAT_EXPECTED(input.Write(part1.data(), part1.size()), Succeeded());
  ASSERT_THAT_EXPECTED(
      Run(/*err=*/nullptr, /*timeout=*/std::chrono::milliseconds(10)),
      HasValue(testing::IsEmpty()));

  ASSERT_THAT_EXPECTED(input.Write(part2.data(), part2.size()), Succeeded());
  input.CloseWriteFileDescriptor();
  ASSERT_THAT_EXPECTED(Run(), HasValue(testing::ElementsAre(Req{"foo"})));
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadWithZeroByteWrites) {
  std::string message = Encode(Req{"foo"});
  auto split_at = message.size() / 2;
  std::string part1 = message.substr(0, split_at);
  std::string part2 = message.substr(split_at);

  ASSERT_THAT_EXPECTED(input.Write(part1.data(), part1.size()), Succeeded());
  ASSERT_THAT_EXPECTED(Run(/*timeout=*/std::chrono::milliseconds(10)),
                       HasValue(testing::IsEmpty()));

  ASSERT_THAT_EXPECTED(input.Write(part1.data(), 0),
                       Succeeded()); // zero-byte write.
  ASSERT_THAT_EXPECTED(Run(/*timeout=*/std::chrono::milliseconds(10)),
                       HasValue(testing::IsEmpty()));

  ASSERT_THAT_EXPECTED(input.Write(part2.data(), part2.size()), Succeeded());
  input.CloseWriteFileDescriptor();
  ASSERT_THAT_EXPECTED(Run(), HasValue(testing::ElementsAre(Req{"foo"})));
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadWithEOF) {
  input.CloseWriteFileDescriptor();
  ASSERT_THAT_EXPECTED(Run(), HasValue(testing::IsEmpty()));
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
  Error err = Error::success();
  ASSERT_THAT_EXPECTED(Run(&err), Succeeded());
  ASSERT_THAT_ERROR(std::move(err), Failed<TransportUnhandledContentsError>());
}

TEST_F(HTTPDelimitedJSONTransportTest, NoDataTimeout) {
  ASSERT_THAT_EXPECTED(Run(/*timeout=*/std::chrono::milliseconds(10)),
                       HasValue(testing::IsEmpty()));
}

TEST_F(HTTPDelimitedJSONTransportTest, InvalidTransport) {
  transport =
      std::make_unique<TestHTTPDelimitedJSONTransport>(nullptr, nullptr);
  ASSERT_THAT_EXPECTED(Run(), FailedWithMessage("IO object is not valid."));
}

TEST_F(HTTPDelimitedJSONTransportTest, Write) {
  ASSERT_THAT_ERROR(transport->Request(Req{"foo"}), Succeeded());
  ASSERT_THAT_ERROR(transport->Response(Resp{"bar"}), Succeeded());
  ASSERT_THAT_ERROR(transport->Event(Evt{"baz"}), Succeeded());
  output.CloseWriteFileDescriptor();
  char buf[1024];
  Expected<size_t> bytes_read =
      output.Read(buf, sizeof(buf), std::chrono::milliseconds(1));
  ASSERT_THAT_EXPECTED(bytes_read, Succeeded());
  ASSERT_EQ(StringRef(buf, *bytes_read), StringRef("Content-Length: 13\r\n\r\n"
                                                   R"({"req":"foo"})"
                                                   "Content-Length: 14\r\n\r\n"
                                                   R"({"resp":"bar"})"
                                                   "Content-Length: 13\r\n\r\n"
                                                   R"({"evt":"baz"})"));
}

TEST_F(JSONRPCTransportTest, MalformedRequests) {
  std::string malformed_header = "notjson\n";
  ASSERT_THAT_EXPECTED(
      input.Write(malformed_header.data(), malformed_header.size()),
      Succeeded());
  Error err = Error::success();
  ASSERT_THAT_EXPECTED(Run(&err), Succeeded());
  ASSERT_THAT_ERROR(std::move(err), FailedWithMessage(testing::HasSubstr(
                                        "Invalid JSON value")));
}

TEST_F(JSONRPCTransportTest, Read) {
  WriteAndCloseInput(Message{Req{"foo"}}, Message{Resp{"bar"}},
                     Message{Evt{"baz"}});
  ASSERT_THAT_EXPECTED(Run(), HasValue(testing::ElementsAre(
                                  Req{"foo"}, Resp{"bar"}, Evt{"baz"})));
}

TEST_F(JSONRPCTransportTest, ReadAcrossMultipleChunks) {
  // Use a string longer than the chunk size to ensure we split the message
  // across the chunk boundary.
  std::string long_str =
      std::string(JSONTransport<Req, Resp, Evt>::kReadBufferSize + 10, 'x');
  WriteAndCloseInput(Req{long_str});
  ASSERT_THAT_EXPECTED(Run(), HasValue(testing::ElementsAre(Req{long_str})));
}

TEST_F(JSONRPCTransportTest, ReadPartialMessage) {
  std::string message = R"({"req": "foo"})"
                        "\n";
  std::string part1 = message.substr(0, 7);
  std::string part2 = message.substr(7);

  ASSERT_THAT_EXPECTED(input.Write(part1.data(), part1.size()), Succeeded());
  ASSERT_THAT_EXPECTED(Run(std::chrono::milliseconds(10)),
                       HasValue(testing::IsEmpty()));

  ASSERT_THAT_EXPECTED(input.Write(part2.data(), part2.size()), Succeeded());
  input.CloseWriteFileDescriptor();
  ASSERT_THAT_EXPECTED(Run(), HasValue(testing::ElementsAre(Req{"foo"})));
}

TEST_F(JSONRPCTransportTest, ReadWithEOF) {
  input.CloseWriteFileDescriptor();
  ASSERT_THAT_EXPECTED(Run(), HasValue(testing::IsEmpty()));
}

TEST_F(JSONRPCTransportTest, ReaderWithUnhandledData) {
  std::string message = R"json({"req": "foo")json";
  // Write an incomplete message and close the handle.
  ASSERT_THAT_EXPECTED(input.Write(message.data(), message.size()),
                       Succeeded());
  input.CloseWriteFileDescriptor();
  Error err = Error::success();
  EXPECT_THAT_EXPECTED(Run(&err), Succeeded());
  ASSERT_THAT_ERROR(std::move(err), Failed<TransportUnhandledContentsError>());
}

TEST_F(JSONRPCTransportTest, Write) {
  ASSERT_THAT_ERROR(transport->Request(Req{"foo"}), Succeeded());
  ASSERT_THAT_ERROR(transport->Response(Resp{"bar"}), Succeeded());
  ASSERT_THAT_ERROR(transport->Event(Evt{"baz"}), Succeeded());
  output.CloseWriteFileDescriptor();
  char buf[1024];
  Expected<size_t> bytes_read =
      output.Read(buf, sizeof(buf), std::chrono::milliseconds(1));
  ASSERT_THAT_EXPECTED(bytes_read, Succeeded());
  ASSERT_EQ(StringRef(buf, *bytes_read), StringRef(R"({"req":"foo"})"
                                                   "\n"
                                                   R"({"resp":"bar"})"
                                                   "\n"
                                                   R"({"evt":"baz"})"
                                                   "\n"));
}

TEST_F(JSONRPCTransportTest, InvalidTransport) {
  transport = std::make_unique<TestJSONRPCTransport>(nullptr, nullptr);
  ASSERT_THAT_EXPECTED(Run(), FailedWithMessage("IO object is not valid."));
}

TEST_F(JSONRPCTransportTest, NoDataTimeout) {
  ASSERT_THAT_EXPECTED(Run(/*timeout=*/std::chrono::milliseconds(10)),
                       HasValue(testing::ElementsAre()));
}

#endif
