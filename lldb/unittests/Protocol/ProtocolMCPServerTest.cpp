//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProtocolMCPTestUtilities.h" // IWYU pragma: keep
#include "TestingSupport/Host/JSONTransportTestUtilities.h"
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
#include <future>
#include <memory>
#include <optional>
#include <system_error>

using namespace llvm;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::transport;
using namespace lldb_protocol::mcp;

// Flakey, see https://github.com/llvm/llvm-project/issues/152677.
#ifndef _WIN32

namespace {

template <typename T> Response make_response(T &&result, Id id = 1) {
  return Response{id, std::forward<T>(result)};
}

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
    return llvm::createStringError(
        std::error_code(eErrorCodeInternalError, std::generic_category()),
        "error");
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

class TestServer : public Server {
public:
  using Server::Bind;
  using Server::Server;
};

using Transport = TestTransport<lldb_protocol::mcp::ProtocolDescriptor>;

class ProtocolServerMCPTest : public testing::Test {
public:
  SubsystemRAII<FileSystem, HostInfo, Socket> subsystems;

  MainLoop loop;
  lldb_private::MainLoop::ReadHandleUP handles[2];

  std::unique_ptr<Transport> to_server;
  MCPBinderUP binder;
  std::unique_ptr<TestServer> server_up;

  std::unique_ptr<Transport> to_client;
  MockMessageHandler<lldb_protocol::mcp::ProtocolDescriptor> client;

  std::vector<std::string> logged_messages;

  /// Runs the MainLoop a single time, executing any pending callbacks.
  void Run() {
    bool addition_succeeded = loop.AddPendingCallback(
        [](MainLoopBase &loop) { loop.RequestTermination(); });
    EXPECT_TRUE(addition_succeeded);
    EXPECT_THAT_ERROR(loop.Run().takeError(), Succeeded());
  }

  void SetUp() override {
    std::tie(to_client, to_server) = Transport::createPair();

    server_up = std::make_unique<TestServer>(
        "lldb-mcp", "0.1.0",
        [this](StringRef msg) { logged_messages.push_back(msg.str()); });
    binder = server_up->Bind(*to_client);
    auto server_handle = to_server->RegisterMessageHandler(loop, *binder);
    EXPECT_THAT_EXPECTED(server_handle, Succeeded());
    binder->OnError([](llvm::Error error) {
      llvm::errs() << formatv("Server transport error: {0}", error);
    });
    handles[0] = std::move(*server_handle);

    auto client_handle = to_client->RegisterMessageHandler(loop, client);
    EXPECT_THAT_EXPECTED(client_handle, Succeeded());
    handles[1] = std::move(*client_handle);
  }

  template <typename Result, typename Params>
  Expected<json::Value> Call(StringRef method, const Params &params) {
    std::promise<Response> promised_result;
    Request req =
        lldb_protocol::mcp::Request{/*id=*/1, method.str(), toJSON(params)};
    EXPECT_THAT_ERROR(to_server->Send(req), Succeeded());
    EXPECT_CALL(client, Received(testing::An<const Response &>()))
        .WillOnce(
            [&](const Response &resp) { promised_result.set_value(resp); });
    Run();
    Response resp = promised_result.get_future().get();
    return toJSON(resp);
  }

  template <typename Result>
  Expected<json::Value>
  Capture(llvm::unique_function<void(Reply<Result>)> &fn) {
    std::promise<llvm::Expected<Result>> promised_result;
    fn([&promised_result](llvm::Expected<Result> result) {
      promised_result.set_value(std::move(result));
    });
    Run();
    llvm::Expected<Result> result = promised_result.get_future().get();
    if (!result)
      return result.takeError();
    return toJSON(*result);
  }

  template <typename Result, typename Params>
  Expected<json::Value>
  Capture(llvm::unique_function<void(const Params &, Reply<Result>)> &fn,
          const Params &params) {
    std::promise<llvm::Expected<Result>> promised_result;
    fn(params, [&promised_result](llvm::Expected<Result> result) {
      promised_result.set_value(std::move(result));
    });
    Run();
    llvm::Expected<Result> result = promised_result.get_future().get();
    if (!result)
      return result.takeError();
    return toJSON(*result);
  }
};

template <typename T>
inline testing::internal::EqMatcher<llvm::json::Value> HasJSON(T x) {
  return testing::internal::EqMatcher<llvm::json::Value>(toJSON(x));
}

} // namespace

TEST_F(ProtocolServerMCPTest, Initialization) {
  EXPECT_THAT_EXPECTED(
      (Call<InitializeResult, InitializeParams>(
          "initialize",
          InitializeParams{/*protocolVersion=*/"2024-11-05",
                           /*capabilities=*/{},
                           /*clientInfo=*/{"lldb-unit", "0.1.0"}})),
      HasValue(make_response(
          InitializeResult{/*protocolVersion=*/"2024-11-05",
                           /*capabilities=*/
                           {
                               /*supportsToolsList=*/true,
                               /*supportsResourcesList=*/true,
                           },
                           /*serverInfo=*/{"lldb-mcp", "0.1.0"}})));
}

TEST_F(ProtocolServerMCPTest, ToolsList) {
  server_up->AddTool(std::make_unique<TestTool>("test", "test tool"));

  ToolDefinition test_tool;
  test_tool.name = "test";
  test_tool.description = "test tool";
  test_tool.inputSchema = json::Object{{"type", "object"}};

  EXPECT_THAT_EXPECTED(Call<ListToolsResult>("tools/list", Void{}),
                       HasValue(make_response(ListToolsResult{{test_tool}})));
}

TEST_F(ProtocolServerMCPTest, ResourcesList) {
  server_up->AddResourceProvider(std::make_unique<TestResourceProvider>());

  EXPECT_THAT_EXPECTED(Call<ListResourcesResult>("resources/list", Void{}),
                       HasValue(make_response(ListResourcesResult{{
                           {
                               /*uri=*/"lldb://foo/bar",
                               /*name=*/"name",
                               /*description=*/"description",
                               /*mimeType=*/"application/json",
                           },
                       }})));
}

TEST_F(ProtocolServerMCPTest, ToolsCall) {
  server_up->AddTool(std::make_unique<TestTool>("test", "test tool"));

  EXPECT_THAT_EXPECTED(
      (Call<CallToolResult, CallToolParams>("tools/call",
                                            CallToolParams{
                                                /*name=*/"test",
                                                /*arguments=*/
                                                json::Object{
                                                    {"arguments", "foo"},
                                                    {"debugger_id", 0},
                                                },
                                            })),
      HasValue(make_response(CallToolResult{{{/*text=*/"foo"}}})));
}

TEST_F(ProtocolServerMCPTest, ToolsCallError) {
  server_up->AddTool(std::make_unique<ErrorTool>("error", "error tool"));

  EXPECT_THAT_EXPECTED((Call<CallToolResult, CallToolParams>(
                           "tools/call", CallToolParams{
                                             /*name=*/"error",
                                             /*arguments=*/
                                             json::Object{
                                                 {"arguments", "foo"},
                                                 {"debugger_id", 0},
                                             },
                                         })),
                       HasValue(make_response(lldb_protocol::mcp::Error{
                           eErrorCodeInternalError, "error"})));
}

TEST_F(ProtocolServerMCPTest, ToolsCallFail) {
  server_up->AddTool(std::make_unique<FailTool>("fail", "fail tool"));

  EXPECT_THAT_EXPECTED((Call<CallToolResult, CallToolParams>(
                           "tools/call", CallToolParams{
                                             /*name=*/"fail",
                                             /*arguments=*/
                                             json::Object{
                                                 {"arguments", "foo"},
                                                 {"debugger_id", 0},
                                             },
                                         })),
                       HasValue(make_response(CallToolResult{
                           {{/*text=*/"failed"}},
                           /*isError=*/true,
                       })));
}

TEST_F(ProtocolServerMCPTest, NotificationInitialized) {
  EXPECT_THAT_ERROR(to_server->Send(lldb_protocol::mcp::Notification{
                        "notifications/initialized",
                        std::nullopt,
                    }),
                    Succeeded());
  Run();
  EXPECT_THAT(logged_messages,
              testing::Contains("MCP initialization complete"));
}

#endif
