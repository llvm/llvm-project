//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestingSupport/Host/JSONTransportTestUtilities.h"
#include "TestingSupport/Host/PipeTestUtilities.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/JSONTransport.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/MainLoopBase.h"
#include "lldb/Host/Socket.h"
#include "lldb/Protocol/MCP/MCPError.h"
#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Resource.h"
#include "lldb/Protocol/MCP/Server.h"
#include "lldb/Protocol/MCP/Tool.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <chrono>
#include <condition_variable>
#include <mutex>

using namespace llvm;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_protocol::mcp;
using testing::_;

namespace {
class TestMCPTransport final : public MCPTransport {
public:
  TestMCPTransport(lldb::IOObjectSP in, lldb::IOObjectSP out)
      : lldb_protocol::mcp::MCPTransport(in, out, "unittest") {}

  using MCPTransport::Write;

  void Log(llvm::StringRef message) override {
    log_messages.emplace_back(message);
  }

  std::vector<std::string> log_messages;
};

class TestServer : public Server {
public:
  using Server::Server;
};

/// Test tool that returns it argument as text.
class TestTool : public Tool {
public:
  using Tool::Tool;

  llvm::Expected<TextResult> Call(const ToolArguments &args) override {
    std::string argument;
    if (const json::Object *args_obj =
            std::get<json::Value>(args).getAsObject()) {
      if (const json::Value *s = args_obj->get("arguments")) {
        argument = s->getAsString().value_or("");
      }
    }

    TextResult text_result;
    text_result.content.emplace_back(TextContent{{argument}});
    return text_result;
  }
};

class TestResourceProvider : public ResourceProvider {
  using ResourceProvider::ResourceProvider;

  std::vector<Resource> GetResources() const override {
    std::vector<Resource> resources;

    Resource resource;
    resource.uri = "lldb://foo/bar";
    resource.name = "name";
    resource.description = "description";
    resource.mimeType = "application/json";

    resources.push_back(resource);
    return resources;
  }

  llvm::Expected<ResourceResult>
  ReadResource(llvm::StringRef uri) const override {
    if (uri != "lldb://foo/bar")
      return llvm::make_error<UnsupportedURI>(uri.str());

    ResourceContents contents;
    contents.uri = "lldb://foo/bar";
    contents.mimeType = "application/json";
    contents.text = "foobar";

    ResourceResult result;
    result.contents.push_back(contents);
    return result;
  }
};

/// Test tool that returns an error.
class ErrorTool : public Tool {
public:
  using Tool::Tool;

  llvm::Expected<TextResult> Call(const ToolArguments &args) override {
    return llvm::createStringError("error");
  }
};

/// Test tool that fails but doesn't return an error.
class FailTool : public Tool {
public:
  using Tool::Tool;

  llvm::Expected<TextResult> Call(const ToolArguments &args) override {
    TextResult text_result;
    text_result.content.emplace_back(TextContent{{"failed"}});
    text_result.isError = true;
    return text_result;
  }
};

class ProtocolServerMCPTest : public PipePairTest {
public:
  SubsystemRAII<FileSystem, HostInfo, Socket> subsystems;

  std::unique_ptr<TestMCPTransport> transport_up;
  std::unique_ptr<TestServer> server_up;
  MainLoop loop;
  MockMessageHandler<Request, Response, Notification> message_handler;

  llvm::Error Write(llvm::StringRef message) {
    llvm::Expected<json::Value> value = json::parse(message);
    if (!value)
      return value.takeError();
    return transport_up->Write(*value);
  }

  /// Run the transport MainLoop and return any messages received.
  llvm::Error
  Run(std::chrono::milliseconds timeout = std::chrono::milliseconds(200)) {
    loop.AddCallback([](MainLoopBase &loop) { loop.RequestTermination(); },
                     timeout);
    auto handle = transport_up->RegisterMessageHandler(loop, message_handler);
    if (!handle)
      return handle.takeError();

    return server_up->Run();
  }

  void SetUp() override {
    PipePairTest::SetUp();

    transport_up = std::make_unique<TestMCPTransport>(
        std::make_shared<NativeFile>(input.GetReadFileDescriptor(),
                                     File::eOpenOptionReadOnly,
                                     NativeFile::Unowned),
        std::make_shared<NativeFile>(output.GetWriteFileDescriptor(),
                                     File::eOpenOptionWriteOnly,
                                     NativeFile::Unowned));

    server_up = std::make_unique<TestServer>(
        "lldb-mcp", "0.1.0",
        std::make_unique<TestMCPTransport>(
            std::make_shared<NativeFile>(output.GetReadFileDescriptor(),
                                         File::eOpenOptionReadOnly,
                                         NativeFile::Unowned),
            std::make_shared<NativeFile>(input.GetWriteFileDescriptor(),
                                         File::eOpenOptionWriteOnly,
                                         NativeFile::Unowned)),
        loop);
  }
};

} // namespace

TEST_F(ProtocolServerMCPTest, Initialization) {
  llvm::StringLiteral request =
      R"json({"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"lldb-unit","version":"0.1.0"}},"jsonrpc":"2.0","id":1})json";
  llvm::StringLiteral response =
      R"json({"id":1,"jsonrpc":"2.0","result":{"capabilities":{"resources":{"listChanged":false,"subscribe":false},"tools":{"listChanged":true}},"protocolVersion":"2024-11-05","serverInfo":{"name":"lldb-mcp","version":"0.1.0"}}})json";

  ASSERT_THAT_ERROR(Write(request), Succeeded());
  llvm::Expected<Response> expected_resp = json::parse<Response>(response);
  ASSERT_THAT_EXPECTED(expected_resp, llvm::Succeeded());
  EXPECT_CALL(message_handler, Received(*expected_resp));
  EXPECT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(ProtocolServerMCPTest, ToolsList) {
  server_up->AddTool(std::make_unique<TestTool>("test", "test tool"));

  llvm::StringLiteral request =
      R"json({"method":"tools/list","params":{},"jsonrpc":"2.0","id":"one"})json";

  ToolDefinition test_tool;
  test_tool.name = "test";
  test_tool.description = "test tool";
  test_tool.inputSchema = json::Object{{"type", "object"}};

  Response response;
  response.id = "one";
  response.result = json::Object{
      {"tools", json::Array{std::move(test_tool)}},
  };

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());
  EXPECT_CALL(message_handler, Received(response));
  EXPECT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(ProtocolServerMCPTest, ResourcesList) {
  server_up->AddResourceProvider(std::make_unique<TestResourceProvider>());

  llvm::StringLiteral request =
      R"json({"method":"resources/list","params":{},"jsonrpc":"2.0","id":2})json";
  llvm::StringLiteral response =
      R"json({"id":2,"jsonrpc":"2.0","result":{"resources":[{"description":"description","mimeType":"application/json","name":"name","uri":"lldb://foo/bar"}]}})json";

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());
  llvm::Expected<Response> expected_resp = json::parse<Response>(response);
  ASSERT_THAT_EXPECTED(expected_resp, llvm::Succeeded());
  EXPECT_CALL(message_handler, Received(*expected_resp));
  EXPECT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(ProtocolServerMCPTest, ToolsCall) {
  server_up->AddTool(std::make_unique<TestTool>("test", "test tool"));

  llvm::StringLiteral request =
      R"json({"method":"tools/call","params":{"name":"test","arguments":{"arguments":"foo","debugger_id":0}},"jsonrpc":"2.0","id":11})json";
  llvm::StringLiteral response =
      R"json({"id":11,"jsonrpc":"2.0","result":{"content":[{"text":"foo","type":"text"}],"isError":false}})json";

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());
  llvm::Expected<Response> expected_resp = json::parse<Response>(response);
  ASSERT_THAT_EXPECTED(expected_resp, llvm::Succeeded());
  EXPECT_CALL(message_handler, Received(*expected_resp));
  EXPECT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(ProtocolServerMCPTest, ToolsCallError) {
  server_up->AddTool(std::make_unique<ErrorTool>("error", "error tool"));

  llvm::StringLiteral request =
      R"json({"method":"tools/call","params":{"name":"error","arguments":{"arguments":"foo","debugger_id":0}},"jsonrpc":"2.0","id":11})json";
  llvm::StringLiteral response =
      R"json({"error":{"code":-32603,"message":"error"},"id":11,"jsonrpc":"2.0"})json";

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());
  llvm::Expected<Response> expected_resp = json::parse<Response>(response);
  ASSERT_THAT_EXPECTED(expected_resp, llvm::Succeeded());
  EXPECT_CALL(message_handler, Received(*expected_resp));
  EXPECT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(ProtocolServerMCPTest, ToolsCallFail) {
  server_up->AddTool(std::make_unique<FailTool>("fail", "fail tool"));

  llvm::StringLiteral request =
      R"json({"method":"tools/call","params":{"name":"fail","arguments":{"arguments":"foo","debugger_id":0}},"jsonrpc":"2.0","id":11})json";
  llvm::StringLiteral response =
      R"json({"id":11,"jsonrpc":"2.0","result":{"content":[{"text":"failed","type":"text"}],"isError":true}})json";

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());
  llvm::Expected<Response> expected_resp = json::parse<Response>(response);
  ASSERT_THAT_EXPECTED(expected_resp, llvm::Succeeded());
  EXPECT_CALL(message_handler, Received(*expected_resp));
  EXPECT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(ProtocolServerMCPTest, NotificationInitialized) {
  bool handler_called = false;
  std::condition_variable cv;

  server_up->AddNotificationHandler(
      "notifications/initialized",
      [&](const Notification &notification) { handler_called = true; });
  llvm::StringLiteral request =
      R"json({"method":"notifications/initialized","jsonrpc":"2.0"})json";

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());
  EXPECT_THAT_ERROR(Run(), Succeeded());
  EXPECT_TRUE(handler_called);
}
