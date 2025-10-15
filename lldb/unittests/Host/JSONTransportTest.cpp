//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/JSONTransport.h"
#include "TestingSupport/Host/JSONTransportTestUtilities.h"
#include "TestingSupport/Host/PipeTestUtilities.h"
#include "TestingSupport/SubsystemRAII.h"
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
#include <optional>
#include <string>
#include <system_error>

using namespace llvm;
using namespace lldb_private;
using namespace lldb_private::transport;
using testing::_;
using testing::HasSubstr;
using testing::InSequence;
using testing::Ref;

namespace llvm::json {
static bool fromJSON(const Value &V, Value &T, Path P) {
  T = V;
  return true;
}
} // namespace llvm::json

namespace {

namespace test_protocol {

struct Request {
  int id = 0;
  std::string name;
  std::optional<json::Value> params;
};
json::Value toJSON(const Request &T) {
  return json::Object{{"name", T.name}, {"id", T.id}, {"params", T.params}};
}
bool fromJSON(const json::Value &V, Request &T, json::Path P) {
  json::ObjectMapper O(V, P);
  return O && O.map("name", T.name) && O.map("id", T.id) &&
         O.map("params", T.params);
}
bool operator==(const Request &a, const Request &b) {
  return a.name == b.name && a.id == b.id && a.params == b.params;
}
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Request &V) {
  OS << toJSON(V);
  return OS;
}
void PrintTo(const Request &message, std::ostream *os) {
  std::string O;
  llvm::raw_string_ostream OS(O);
  OS << message;
  *os << O;
}

struct Response {
  int id = 0;
  int errorCode = 0;
  std::optional<json::Value> result;
};
json::Value toJSON(const Response &T) {
  return json::Object{
      {"id", T.id}, {"errorCode", T.errorCode}, {"result", T.result}};
}
bool fromJSON(const json::Value &V, Response &T, json::Path P) {
  json::ObjectMapper O(V, P);
  return O && O.map("id", T.id) && O.mapOptional("errorCode", T.errorCode) &&
         O.map("result", T.result);
}
bool operator==(const Response &a, const Response &b) {
  return a.id == b.id && a.errorCode == b.errorCode && a.result == b.result;
}
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Response &V) {
  OS << toJSON(V);
  return OS;
}
void PrintTo(const Response &message, std::ostream *os) {
  std::string O;
  llvm::raw_string_ostream OS(O);
  OS << message;
  *os << O;
}

struct Event {
  std::string name;
  std::optional<json::Value> params;
};
json::Value toJSON(const Event &T) {
  return json::Object{{"name", T.name}, {"params", T.params}};
}
bool fromJSON(const json::Value &V, Event &T, json::Path P) {
  json::ObjectMapper O(V, P);
  return O && O.map("name", T.name) && O.map("params", T.params);
}
bool operator==(const Event &a, const Event &b) { return a.name == b.name; }
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Event &V) {
  OS << toJSON(V);
  return OS;
}
void PrintTo(const Event &message, std::ostream *os) {
  std::string O;
  llvm::raw_string_ostream OS(O);
  OS << message;
  *os << O;
}

using Message = std::variant<Request, Response, Event>;
json::Value toJSON(const Message &msg) {
  return std::visit([](const auto &msg) { return toJSON(msg); }, msg);
}
bool fromJSON(const json::Value &V, Message &msg, json::Path P) {
  const json::Object *O = V.getAsObject();
  if (!O) {
    P.report("expected object");
    return false;
  }

  if (O->find("id") == O->end()) {
    Event E;
    if (!fromJSON(V, E, P))
      return false;

    msg = std::move(E);
    return true;
  }

  if (O->get("name")) {
    Request R;
    if (!fromJSON(V, R, P))
      return false;

    msg = std::move(R);
    return true;
  }

  Response R;
  if (!fromJSON(V, R, P))
    return false;

  msg = std::move(R);
  return true;
}

struct MyFnParams {
  int a = 0;
  int b = 0;
};
json::Value toJSON(const MyFnParams &T) {
  return json::Object{{"a", T.a}, {"b", T.b}};
}
bool fromJSON(const json::Value &V, MyFnParams &T, json::Path P) {
  json::ObjectMapper O(V, P);
  return O && O.map("a", T.a) && O.map("b", T.b);
}

struct MyFnResult {
  int c = 0;
};
json::Value toJSON(const MyFnResult &T) { return json::Object{{"c", T.c}}; }
bool fromJSON(const json::Value &V, MyFnResult &T, json::Path P) {
  json::ObjectMapper O(V, P);
  return O && O.map("c", T.c);
}

struct ProtoDesc {
  using Id = int;
  using Req = Request;
  using Resp = Response;
  using Evt = Event;

  static inline Id InitialId() { return 0; }
  static inline Req Make(Id id, llvm::StringRef method,
                         std::optional<llvm::json::Value> params) {
    return Req{id, method.str(), params};
  }
  static inline Evt Make(llvm::StringRef method,
                         std::optional<llvm::json::Value> params) {
    return Evt{method.str(), params};
  }
  static inline Resp Make(Req req, llvm::Error error) {
    Resp resp;
    resp.id = req.id;
    llvm::handleAllErrors(
        std::move(error), [&](const llvm::ErrorInfoBase &err) {
          std::error_code cerr = err.convertToErrorCode();
          resp.errorCode =
              cerr == llvm::inconvertibleErrorCode() ? 1 : cerr.value();
          resp.result = err.message();
        });
    return resp;
  }
  static inline Resp Make(Req req, std::optional<llvm::json::Value> result) {
    return Resp{req.id, 0, std::move(result)};
  }
  static inline Id KeyFor(Resp r) { return r.id; }
  static inline std::string KeyFor(Req r) { return r.name; }
  static inline std::string KeyFor(Evt e) { return e.name; }
  static inline std::optional<llvm::json::Value> Extract(Req r) {
    return r.params;
  }
  static inline llvm::Expected<llvm::json::Value> Extract(Resp r) {
    if (r.errorCode != 0)
      return llvm::createStringError(
          std::error_code(r.errorCode, std::generic_category()),
          r.result && r.result->getAsString() ? *r.result->getAsString()
                                              : "no-message");
    return r.result;
  }
  static inline std::optional<llvm::json::Value> Extract(Evt e) {
    return e.params;
  }
};

using Transport = TestTransport<ProtoDesc>;
using Binder = lldb_private::transport::Binder<ProtoDesc>;
using MessageHandler = MockMessageHandler<ProtoDesc>;

} // namespace test_protocol

template <typename T> class JSONTransportTest : public PipePairTest {
protected:
  SubsystemRAII<FileSystem> subsystems;

  test_protocol::MessageHandler message_handler;
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

  /// Run the transport MainLoop and return any messages received.
  Error
  Run(bool close_input = true,
      std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
    if (close_input) {
      input.CloseWriteFileDescriptor();
      EXPECT_CALL(message_handler, OnClosed()).WillOnce([this]() {
        loop.RequestTermination();
      });
    }
    loop.AddCallback(
        [](MainLoopBase &loop) {
          loop.RequestTermination();
          FAIL() << "timeout";
        },
        timeout);
    auto handle = transport->RegisterMessageHandler(loop, message_handler);
    if (!handle)
      return handle.takeError();

    return loop.Run().takeError();
  }

  template <typename... Ts> void Write(Ts... args) {
    std::string message;
    for (const auto &arg : {args...})
      message += Encode(arg);
    EXPECT_THAT_EXPECTED(input.Write(message.data(), message.size()),
                         Succeeded());
  }

  virtual std::string Encode(const json::Value &) = 0;
};

class TestHTTPDelimitedJSONTransport final
    : public HTTPDelimitedJSONTransport<test_protocol::ProtoDesc> {
public:
  using HTTPDelimitedJSONTransport::HTTPDelimitedJSONTransport;

  void Log(llvm::StringRef message) override {
    log_messages.emplace_back(message);
  }

  std::vector<std::string> log_messages;
};

class HTTPDelimitedJSONTransportTest
    : public JSONTransportTest<TestHTTPDelimitedJSONTransport> {
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
    : public JSONRPCTransport<test_protocol::ProtoDesc> {
public:
  using JSONRPCTransport::JSONRPCTransport;

  void Log(llvm::StringRef message) override {
    log_messages.emplace_back(message);
  }

  std::vector<std::string> log_messages;
};

class JSONRPCTransportTest : public JSONTransportTest<TestJSONRPCTransport> {
public:
  using JSONTransportTest::JSONTransportTest;

  std::string Encode(const json::Value &V) override {
    std::string msg;
    raw_string_ostream OS(msg);
    OS << formatv("{0}\n", V);
    return msg;
  }
};

class TransportBinderTest : public testing::Test {
protected:
  SubsystemRAII<FileSystem> subsystems;

  std::unique_ptr<test_protocol::Transport> to_remote;
  std::unique_ptr<test_protocol::Transport> from_remote;
  std::unique_ptr<test_protocol::Binder> binder;
  test_protocol::MessageHandler remote;
  MainLoop loop;

  void SetUp() override {
    std::tie(to_remote, from_remote) = test_protocol::Transport::createPair();
    binder = std::make_unique<test_protocol::Binder>(*to_remote);

    auto binder_handle = to_remote->RegisterMessageHandler(loop, remote);
    EXPECT_THAT_EXPECTED(binder_handle, Succeeded());

    auto remote_handle = from_remote->RegisterMessageHandler(loop, *binder);
    EXPECT_THAT_EXPECTED(remote_handle, Succeeded());
  }

  void Run() {
    loop.AddPendingCallback([](auto &loop) { loop.RequestTermination(); });
    EXPECT_THAT_ERROR(loop.Run().takeError(), Succeeded());
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

  EXPECT_CALL(message_handler, OnError(_)).WillOnce([](llvm::Error err) {
    ASSERT_THAT_ERROR(std::move(err),
                      FailedWithMessage("invalid content length: -1"));
  });
  ASSERT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(HTTPDelimitedJSONTransportTest, Read) {
  Write(Request{6, "foo", std::nullopt});
  EXPECT_CALL(message_handler, Received(Request{6, "foo", std::nullopt}));
  ASSERT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadMultipleMessagesInSingleWrite) {
  InSequence seq;
  Write(
      Message{
          Request{6, "one", std::nullopt},
      },
      Message{
          test_protocol::Event{"two", std::nullopt},
      },
      Message{
          Response{2, 0, std::nullopt},
      });
  EXPECT_CALL(message_handler, Received(Request{6, "one", std::nullopt}));
  EXPECT_CALL(message_handler,
              Received(test_protocol::Event{"two", std::nullopt}));
  EXPECT_CALL(message_handler, Received(Response{2, 0, std::nullopt}));
  ASSERT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadAcrossMultipleChunks) {
  std::string long_str = std::string(
      HTTPDelimitedJSONTransport<test_protocol::ProtoDesc>::kReadBufferSize * 2,
      'x');
  Write(Request{5, long_str, std::nullopt});
  EXPECT_CALL(message_handler, Received(Request{5, long_str, std::nullopt}));
  ASSERT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadPartialMessage) {
  std::string message = Encode(Request{5, "foo", std::nullopt});
  auto split_at = message.size() / 2;
  std::string part1 = message.substr(0, split_at);
  std::string part2 = message.substr(split_at);

  EXPECT_CALL(message_handler, Received(Request{5, "foo", std::nullopt}));

  ASSERT_THAT_EXPECTED(input.Write(part1.data(), part1.size()), Succeeded());
  loop.AddPendingCallback(
      [](MainLoopBase &loop) { loop.RequestTermination(); });
  ASSERT_THAT_ERROR(Run(/*close_stdin=*/false), Succeeded());
  ASSERT_THAT_EXPECTED(input.Write(part2.data(), part2.size()), Succeeded());
  input.CloseWriteFileDescriptor();
  ASSERT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadWithZeroByteWrites) {
  std::string message = Encode(Request{6, "foo", std::nullopt});
  auto split_at = message.size() / 2;
  std::string part1 = message.substr(0, split_at);
  std::string part2 = message.substr(split_at);

  EXPECT_CALL(message_handler, Received(Request{6, "foo", std::nullopt}));

  ASSERT_THAT_EXPECTED(input.Write(part1.data(), part1.size()), Succeeded());

  // Run the main loop once for the initial read.
  loop.AddPendingCallback(
      [](MainLoopBase &loop) { loop.RequestTermination(); });
  ASSERT_THAT_ERROR(Run(/*close_stdin=*/false), Succeeded());

  // zero-byte write.
  ASSERT_THAT_EXPECTED(input.Write(part1.data(), 0),
                       Succeeded()); // zero-byte write.
  loop.AddPendingCallback(
      [](MainLoopBase &loop) { loop.RequestTermination(); });
  ASSERT_THAT_ERROR(Run(/*close_stdin=*/false), Succeeded());

  // Write the remaining part of the message.
  ASSERT_THAT_EXPECTED(input.Write(part2.data(), part2.size()), Succeeded());
  ASSERT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadWithEOF) {
  ASSERT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(HTTPDelimitedJSONTransportTest, ReaderWithUnhandledData) {
  std::string json = R"json({"str": "foo"})json";
  std::string message =
      formatv("Content-Length: {0}\r\nContent-type: text/json\r\n\r\n{1}",
              json.size(), json)
          .str();

  EXPECT_CALL(message_handler, OnError(_)).WillOnce([](llvm::Error err) {
    // The error should indicate that there are unhandled contents.
    ASSERT_THAT_ERROR(std::move(err),
                      Failed<TransportUnhandledContentsError>());
  });

  // Write an incomplete message and close the handle.
  ASSERT_THAT_EXPECTED(input.Write(message.data(), message.size() - 1),
                       Succeeded());
  ASSERT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(HTTPDelimitedJSONTransportTest, InvalidTransport) {
  transport =
      std::make_unique<TestHTTPDelimitedJSONTransport>(nullptr, nullptr);
  ASSERT_THAT_ERROR(Run(/*close_input=*/false),
                    FailedWithMessage("IO object is not valid."));
}

TEST_F(HTTPDelimitedJSONTransportTest, Write) {
  ASSERT_THAT_ERROR(transport->Send(Request{7, "foo", std::nullopt}),
                    Succeeded());
  ASSERT_THAT_ERROR(transport->Send(Response{5, 0, "bar"}), Succeeded());
  ASSERT_THAT_ERROR(transport->Send(test_protocol::Event{"baz", std::nullopt}),
                    Succeeded());
  output.CloseWriteFileDescriptor();
  char buf[1024];
  Expected<size_t> bytes_read =
      output.Read(buf, sizeof(buf), std::chrono::milliseconds(1));
  ASSERT_THAT_EXPECTED(bytes_read, Succeeded());
  ASSERT_EQ(StringRef(buf, *bytes_read),
            StringRef("Content-Length: 35\r\n\r\n"
                      R"({"id":7,"name":"foo","params":null})"
                      "Content-Length: 37\r\n\r\n"
                      R"({"errorCode":0,"id":5,"result":"bar"})"
                      "Content-Length: 28\r\n\r\n"
                      R"({"name":"baz","params":null})"));
}

TEST_F(JSONRPCTransportTest, MalformedRequests) {
  std::string malformed_header = "notjson\n";
  ASSERT_THAT_EXPECTED(
      input.Write(malformed_header.data(), malformed_header.size()),
      Succeeded());
  EXPECT_CALL(message_handler, OnError(_)).WillOnce([](llvm::Error err) {
    ASSERT_THAT_ERROR(std::move(err),
                      FailedWithMessage(HasSubstr("Invalid JSON value")));
  });
  ASSERT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(JSONRPCTransportTest, Read) {
  Write(Message{Request{1, "foo", std::nullopt}});
  EXPECT_CALL(message_handler, Received(Request{1, "foo", std::nullopt}));
  ASSERT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(JSONRPCTransportTest, ReadMultipleMessagesInSingleWrite) {
  InSequence seq;
  Write(Message{Request{1, "one", std::nullopt}},
        Message{test_protocol::Event{"two", std::nullopt}},
        Message{Response{3, 0, "three"}});
  EXPECT_CALL(message_handler, Received(Request{1, "one", std::nullopt}));
  EXPECT_CALL(message_handler,
              Received(test_protocol::Event{"two", std::nullopt}));
  EXPECT_CALL(message_handler, Received(Response{3, 0, "three"}));
  ASSERT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(JSONRPCTransportTest, ReadAcrossMultipleChunks) {
  // Use a string longer than the chunk size to ensure we split the message
  // across the chunk boundary.
  std::string long_str = std::string(
      IOTransport<test_protocol::ProtoDesc>::kReadBufferSize * 2, 'x');
  Write(Request{42, long_str, std::nullopt});
  EXPECT_CALL(message_handler, Received(Request{42, long_str, std::nullopt}));
  ASSERT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(JSONRPCTransportTest, ReadPartialMessage) {
  std::string message = R"({"id":42,"name":"foo","params":null})"
                        "\n";
  std::string part1 = message.substr(0, 7);
  std::string part2 = message.substr(7);

  EXPECT_CALL(message_handler, Received(Request{42, "foo", std::nullopt}));

  ASSERT_THAT_EXPECTED(input.Write(part1.data(), part1.size()), Succeeded());
  loop.AddPendingCallback(
      [](MainLoopBase &loop) { loop.RequestTermination(); });
  ASSERT_THAT_ERROR(Run(/*close_input=*/false), Succeeded());

  ASSERT_THAT_EXPECTED(input.Write(part2.data(), part2.size()), Succeeded());
  input.CloseWriteFileDescriptor();
  ASSERT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(JSONRPCTransportTest, ReadWithEOF) {
  ASSERT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(JSONRPCTransportTest, ReaderWithUnhandledData) {
  std::string message = R"json({"req": "foo")json";
  // Write an incomplete message and close the handle.
  ASSERT_THAT_EXPECTED(input.Write(message.data(), message.size() - 1),
                       Succeeded());

  EXPECT_CALL(message_handler, OnError(_)).WillOnce([](llvm::Error err) {
    ASSERT_THAT_ERROR(std::move(err),
                      Failed<TransportUnhandledContentsError>());
  });
  ASSERT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(JSONRPCTransportTest, Write) {
  ASSERT_THAT_ERROR(transport->Send(Request{11, "foo", std::nullopt}),
                    Succeeded());
  ASSERT_THAT_ERROR(transport->Send(Response{14, 0, "bar"}), Succeeded());
  ASSERT_THAT_ERROR(transport->Send(test_protocol::Event{"baz", std::nullopt}),
                    Succeeded());
  output.CloseWriteFileDescriptor();
  char buf[1024];
  Expected<size_t> bytes_read =
      output.Read(buf, sizeof(buf), std::chrono::milliseconds(1));
  ASSERT_THAT_EXPECTED(bytes_read, Succeeded());
  ASSERT_EQ(StringRef(buf, *bytes_read),
            StringRef(R"({"id":11,"name":"foo","params":null})"
                      "\n"
                      R"({"errorCode":0,"id":14,"result":"bar"})"
                      "\n"
                      R"({"name":"baz","params":null})"
                      "\n"));
}

TEST_F(JSONRPCTransportTest, InvalidTransport) {
  transport = std::make_unique<TestJSONRPCTransport>(nullptr, nullptr);
  ASSERT_THAT_ERROR(Run(/*close_input=*/false),
                    FailedWithMessage("IO object is not valid."));
}

// Out-bound binding request handler.
TEST_F(TransportBinderTest, OutBoundRequests) {
  OutgoingRequest<MyFnResult, MyFnParams> addFn =
      binder->Bind<MyFnResult, MyFnParams>("add");
  bool replied = false;
  addFn(MyFnParams{1, 2}, [&](Expected<MyFnResult> result) {
    EXPECT_THAT_EXPECTED(result, Succeeded());
    EXPECT_EQ(result->c, 3);
    replied = true;
  });
  EXPECT_CALL(remote, Received(Request{1, "add", MyFnParams{1, 2}}));
  EXPECT_THAT_ERROR(from_remote->Send(Response{1, 0, toJSON(MyFnResult{3})}),
                    Succeeded());
  Run();
  EXPECT_TRUE(replied);
}

TEST_F(TransportBinderTest, OutBoundRequestsVoidParams) {
  OutgoingRequest<MyFnResult, void> voidParamFn =
      binder->Bind<MyFnResult, void>("voidParam");
  bool replied = false;
  voidParamFn([&](Expected<MyFnResult> result) {
    EXPECT_THAT_EXPECTED(result, Succeeded());
    EXPECT_EQ(result->c, 3);
    replied = true;
  });
  EXPECT_CALL(remote, Received(Request{1, "voidParam", std::nullopt}));
  EXPECT_THAT_ERROR(from_remote->Send(Response{1, 0, toJSON(MyFnResult{3})}),
                    Succeeded());
  Run();
  EXPECT_TRUE(replied);
}

TEST_F(TransportBinderTest, OutBoundRequestsVoidResult) {
  OutgoingRequest<void, MyFnParams> voidResultFn =
      binder->Bind<void, MyFnParams>("voidResult");
  bool replied = false;
  voidResultFn(MyFnParams{4, 5}, [&](llvm::Error error) {
    EXPECT_THAT_ERROR(std::move(error), Succeeded());
    replied = true;
  });
  EXPECT_CALL(remote, Received(Request{1, "voidResult", MyFnParams{4, 5}}));
  EXPECT_THAT_ERROR(from_remote->Send(Response{1, 0, std::nullopt}),
                    Succeeded());
  Run();
  EXPECT_TRUE(replied);
}

TEST_F(TransportBinderTest, OutBoundRequestsVoidParamsAndVoidResult) {
  OutgoingRequest<void, void> voidParamAndResultFn =
      binder->Bind<void, void>("voidParamAndResult");
  bool replied = false;
  voidParamAndResultFn([&](llvm::Error error) {
    EXPECT_THAT_ERROR(std::move(error), Succeeded());
    replied = true;
  });
  EXPECT_CALL(remote, Received(Request{1, "voidParamAndResult", std::nullopt}));
  EXPECT_THAT_ERROR(from_remote->Send(Response{1, 0, std::nullopt}),
                    Succeeded());
  Run();
  EXPECT_TRUE(replied);
}

// In-bound binding request handler.
TEST_F(TransportBinderTest, InBoundRequests) {
  bool called = false;
  binder->Bind<MyFnResult, MyFnParams>(
      "add",
      [&](const int captured_param,
          const MyFnParams &params) -> Expected<MyFnResult> {
        called = true;
        return MyFnResult{params.a + params.b + captured_param};
      },
      2);
  EXPECT_THAT_ERROR(from_remote->Send(Request{1, "add", MyFnParams{3, 4}}),
                    Succeeded());

  EXPECT_CALL(remote, Received(Response{1, 0, MyFnResult{9}}));
  Run();
  EXPECT_TRUE(called);
}

TEST_F(TransportBinderTest, InBoundRequestsVoidParams) {
  bool called = false;
  binder->Bind<MyFnResult, void>(
      "voidParam",
      [&](const int captured_param) -> Expected<MyFnResult> {
        called = true;
        return MyFnResult{captured_param};
      },
      2);
  EXPECT_THAT_ERROR(from_remote->Send(Request{2, "voidParam", std::nullopt}),
                    Succeeded());
  EXPECT_CALL(remote, Received(Response{2, 0, MyFnResult{2}}));
  Run();
  EXPECT_TRUE(called);
}

TEST_F(TransportBinderTest, InBoundRequestsVoidResult) {
  bool called = false;
  binder->Bind<void, MyFnParams>(
      "voidResult",
      [&](const int captured_param, const MyFnParams &params) -> llvm::Error {
        called = true;
        EXPECT_EQ(captured_param, 2);
        EXPECT_EQ(params.a, 3);
        EXPECT_EQ(params.b, 4);
        return llvm::Error::success();
      },
      2);
  EXPECT_THAT_ERROR(
      from_remote->Send(Request{3, "voidResult", MyFnParams{3, 4}}),
      Succeeded());
  EXPECT_CALL(remote, Received(Response{3, 0, std::nullopt}));
  Run();
  EXPECT_TRUE(called);
}
TEST_F(TransportBinderTest, InBoundRequestsVoidParamsAndResult) {
  bool called = false;
  binder->Bind<void, void>(
      "voidParamAndResult",
      [&](const int captured_param) -> llvm::Error {
        called = true;
        EXPECT_EQ(captured_param, 2);
        return llvm::Error::success();
      },
      2);
  EXPECT_THAT_ERROR(
      from_remote->Send(Request{4, "voidParamAndResult", std::nullopt}),
      Succeeded());
  EXPECT_CALL(remote, Received(Response{4, 0, std::nullopt}));
  Run();
  EXPECT_TRUE(called);
}

// Out-bound binding event handler.
TEST_F(TransportBinderTest, OutBoundEvents) {
  OutgoingEvent<MyFnParams> emitEvent = binder->Bind<MyFnParams>("evt");
  emitEvent(MyFnParams{1, 2});
  EXPECT_CALL(remote, Received(test_protocol::Event{"evt", MyFnParams{1, 2}}));
  Run();
}

TEST_F(TransportBinderTest, OutBoundEventsVoidParams) {
  OutgoingEvent<void> emitEvent = binder->Bind<void>("evt");
  emitEvent();
  EXPECT_CALL(remote, Received(test_protocol::Event{"evt", std::nullopt}));
  Run();
}

// In-bound binding event handler.
TEST_F(TransportBinderTest, InBoundEvents) {
  bool called = false;
  binder->Bind<MyFnParams>(
      "evt",
      [&](const int captured_arg, const MyFnParams &params) {
        EXPECT_EQ(captured_arg, 42);
        EXPECT_EQ(params.a, 3);
        EXPECT_EQ(params.b, 4);
        called = true;
      },
      42);
  EXPECT_THAT_ERROR(
      from_remote->Send(test_protocol::Event{"evt", MyFnParams{3, 4}}),
      Succeeded());
  Run();
  EXPECT_TRUE(called);
}

TEST_F(TransportBinderTest, InBoundEventsVoidParams) {
  bool called = false;
  binder->Bind<void>(
      "evt",
      [&](const int captured_arg) {
        EXPECT_EQ(captured_arg, 42);
        called = true;
      },
      42);
  EXPECT_THAT_ERROR(
      from_remote->Send(test_protocol::Event{"evt", std::nullopt}),
      Succeeded());
  Run();
  EXPECT_TRUE(called);
}

#endif
