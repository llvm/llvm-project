//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProtocolMCPTestUtilities.h"
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
#include "lldb/Protocol/MCP/Transport.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <chrono>
#include <condition_variable>

using namespace llvm;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_protocol::mcp;

namespace {

class TestServer : public Server {
public:
  using Server::Server;
};

/// Test tool that returns it argument as text.
class TestTool : public Tool {
public:
  using Tool::Tool;

  llvm::Expected<CallToolResult> Call(const ToolArguments &args) override {
    std::string argument;
    if (const json::Object *args_obj =
            std::get<json::Value>(args).getAsObject()) {
      if (const json::Value *s = args_obj->get("arguments")) {
        argument = s->getAsString().value_or("");
      }
    }

    CallToolResult text_result;
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

  llvm::Expected<ReadResourceResult>
  ReadResource(llvm::StringRef uri) const override {
    if (uri != "lldb://foo/bar")
      return llvm::make_error<UnsupportedURI>(uri.str());

    TextResourceContents contents;
    contents.uri = "lldb://foo/bar";
    contents.mimeType = "application/json";
    contents.text = "foobar";

    ReadResourceResult result;
    result.contents.push_back(contents);
    return result;
  }
};

/// Test tool that returns an error.
class ErrorTool : public Tool {
public:
  using Tool::Tool;

  llvm::Expected<CallToolResult> Call(const ToolArguments &args) override {
    return llvm::createStringError("error");
  }
};

/// Test tool that fails but doesn't return an error.
class FailTool : public Tool {
public:
  using Tool::Tool;

  llvm::Expected<CallToolResult> Call(const ToolArguments &args) override {
    CallToolResult text_result;
    text_result.content.emplace_back(TextContent{{"failed"}});
    text_result.isError = true;
    return text_result;
  }
};

class ProtocolServerMCPTest : public PipePairTest {
public:
  SubsystemRAII<FileSystem, HostInfo, Socket> subsystems;

  std::unique_ptr<lldb_protocol::mcp::Transport> transport_up;
  std::unique_ptr<TestServer> server_up;
  MainLoop loop;
  MockMessageHandler<Request, Response, Notification> message_handler;

  llvm::Error Write(llvm::StringRef message) {
    llvm::Expected<json::Value> value = json::parse(message);
    if (!value)
      return value.takeError();
    return transport_up->Write(*value);
  }

  llvm::Error Write(json::Value value) { return transport_up->Write(value); }

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

    transport_up = std::make_unique<lldb_protocol::mcp::Transport>(
        std::make_shared<NativeFile>(input.GetReadFileDescriptor(),
                                     File::eOpenOptionReadOnly,
                                     NativeFile::Unowned),
        std::make_shared<NativeFile>(output.GetWriteFileDescriptor(),
                                     File::eOpenOptionWriteOnly,
                                     NativeFile::Unowned));

    server_up = std::make_unique<TestServer>(
        "lldb-mcp", "0.1.0",
        std::make_unique<lldb_protocol::mcp::Transport>(
            std::make_shared<NativeFile>(output.GetReadFileDescriptor(),
                                         File::eOpenOptionReadOnly,
                                         NativeFile::Unowned),
            std::make_shared<NativeFile>(input.GetWriteFileDescriptor(),
                                         File::eOpenOptionWriteOnly,
                                         NativeFile::Unowned)),
        loop);
  }
};

template <typename T>
Request make_request(StringLiteral method, T &&params, Id id = 1) {
  return Request{id, method.str(), toJSON(std::forward<T>(params))};
}

template <typename T> Response make_response(T &&result, Id id = 1) {
  return Response{id, std::forward<T>(result)};
}

} // namespace

TEST_F(ProtocolServerMCPTest, Initialization) {
  Request request = make_request(
      "initialize", InitializeParams{/*protocolVersion=*/"2024-11-05",
                                     /*capabilities=*/{},
                                     /*clientInfo=*/{"lldb-unit", "0.1.0"}});
  Response response = make_response(
      InitializeResult{/*protocolVersion=*/"2024-11-05",
                       /*capabilities=*/{/*supportsToolsList=*/true},
                       /*serverInfo=*/{"lldb-mcp", "0.1.0"}});

  ASSERT_THAT_ERROR(Write(request), Succeeded());
  EXPECT_CALL(message_handler, Received(response));
  EXPECT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(ProtocolServerMCPTest, ToolsList) {
  server_up->AddTool(std::make_unique<TestTool>("test", "test tool"));

  Request request = make_request("tools/list", Void{}, /*id=*/"one");

  ToolDefinition test_tool;
  test_tool.name = "test";
  test_tool.description = "test tool";
  test_tool.inputSchema = json::Object{{"type", "object"}};

  Response response = make_response(ListToolsResult{{test_tool}}, /*id=*/"one");

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());
  EXPECT_CALL(message_handler, Received(response));
  EXPECT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(ProtocolServerMCPTest, ResourcesList) {
  server_up->AddResourceProvider(std::make_unique<TestResourceProvider>());

  Request request = make_request("resources/list", Void{});
  Response response = make_response(ListResourcesResult{
      {{/*uri=*/"lldb://foo/bar", /*name=*/"name",
        /*description=*/"description", /*mimeType=*/"application/json"}}});

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());
  EXPECT_CALL(message_handler, Received(response));
  EXPECT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(ProtocolServerMCPTest, ToolsCall) {
  server_up->AddTool(std::make_unique<TestTool>("test", "test tool"));

  Request request = make_request(
      "tools/call", CallToolParams{/*name=*/"test", /*arguments=*/json::Object{
                                       {"arguments", "foo"},
                                       {"debugger_id", 0},
                                   }});
  Response response = make_response(CallToolResult{{{/*text=*/"foo"}}});

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());
  EXPECT_CALL(message_handler, Received(response));
  EXPECT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(ProtocolServerMCPTest, ToolsCallError) {
  server_up->AddTool(std::make_unique<ErrorTool>("error", "error tool"));

  Request request = make_request(
      "tools/call", CallToolParams{/*name=*/"error", /*arguments=*/json::Object{
                                       {"arguments", "foo"},
                                       {"debugger_id", 0},
                                   }});
  Response response =
      make_response(lldb_protocol::mcp::Error{eErrorCodeInternalError,
                                              /*message=*/"error"});

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());
  EXPECT_CALL(message_handler, Received(response));
  EXPECT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(ProtocolServerMCPTest, ToolsCallFail) {
  server_up->AddTool(std::make_unique<FailTool>("fail", "fail tool"));

  Request request = make_request(
      "tools/call", CallToolParams{/*name=*/"fail", /*arguments=*/json::Object{
                                       {"arguments", "foo"},
                                       {"debugger_id", 0},
                                   }});
  Response response =
      make_response(CallToolResult{{{/*text=*/"failed"}}, /*isError=*/true});

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());
  EXPECT_CALL(message_handler, Received(response));
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
