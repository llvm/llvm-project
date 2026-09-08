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
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/MainLoopBase.h"
#include "lldb/Host/Socket.h"
#include "lldb/Protocol/MCP/Client.h"
#include "lldb/Protocol/MCP/MCPError.h"
#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Resource.h"
#include "lldb/Protocol/MCP/Server.h"
#include "lldb/Protocol/MCP/Tool.h"
#include "lldb/Protocol/MCP/Transport.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <future>
#include <memory>

using namespace llvm;
using namespace lldb_private;
using namespace lldb_private::transport;
using namespace lldb_protocol::mcp;

// Flakey, see https://github.com/llvm/llvm-project/issues/152677.
#ifndef _WIN32

namespace {

/// A tool that echoes a fixed string.
class EchoTool : public Tool {
public:
  using Tool::Tool;

  llvm::Expected<CallToolResult> Call(const ToolArguments &) override {
    CallToolResult result;
    result.content.emplace_back(TextContent{{"echo"}});
    return result;
  }
};

/// A tool that fails with an MCP error.
class ErrorTool : public Tool {
public:
  using Tool::Tool;

  llvm::Expected<CallToolResult> Call(const ToolArguments &) override {
    return llvm::createStringError("boom");
  }
};

class TestResourceProvider : public ResourceProvider {
public:
  using ResourceProvider::ResourceProvider;

  std::vector<Resource> GetResources() const override {
    Resource resource;
    resource.uri = "lldb://foo/bar";
    resource.name = "name";
    resource.description = "description";
    resource.mimeType = "application/json";
    return {resource};
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

class MCPClientTest : public testing::Test {
public:
  SubsystemRAII<FileSystem, HostInfo, Socket> subsystems;

  MainLoop loop;
  std::unique_ptr<Server> server;
  std::unique_ptr<Client> client;
  TestTransport<ProtocolDescriptor> *client_transport = nullptr;

  void SetUp() override {
    auto pair = TestTransport<ProtocolDescriptor>::createConnectedPair(loop);
    client_transport = pair.first.get();

    server = std::make_unique<Server>("lldb-mcp", "0.1.0");
    server->AddTool(std::make_unique<EchoTool>("echo", "echo tool"));
    server->AddTool(std::make_unique<ErrorTool>("error", "error tool"));
    server->AddResourceProvider(std::make_unique<TestResourceProvider>());
    EXPECT_THAT_ERROR(server->Accept(std::move(pair.second)), Succeeded());

    client = std::make_unique<Client>(std::move(pair.first));
    EXPECT_THAT_ERROR(client->Run(), Succeeded());
  }

  /// Runs the MainLoop, draining pending callbacks.
  void Run() {
    bool addition_succeeded = loop.AddPendingCallback(
        [](MainLoopBase &loop) { loop.RequestTermination(); });
    EXPECT_TRUE(addition_succeeded);
    EXPECT_THAT_ERROR(loop.Run().takeError(), Succeeded());
  }

  /// Issues an asynchronous request and returns its captured result.
  template <typename Result>
  llvm::Expected<Result>
  Capture(llvm::unique_function<void(Client::Reply<Result>)> invoke) {
    std::promise<llvm::Expected<Result>> promised_result;
    invoke([&promised_result](llvm::Expected<Result> result) {
      promised_result.set_value(std::move(result));
    });
    Run();
    return promised_result.get_future().get();
  }
};

} // namespace

TEST_F(MCPClientTest, Initialize) {
  llvm::Expected<InitializeResult> result =
      Capture<InitializeResult>([&](Client::Reply<InitializeResult> reply) {
        client->Initialize(
            InitializeParams{/*protocolVersion=*/"2024-11-05",
                             /*capabilities=*/{},
                             /*clientInfo=*/{"lldb-unit", "0.1.0"}},
            std::move(reply));
      });
  ASSERT_THAT_EXPECTED(result, Succeeded());
  EXPECT_EQ(result->protocolVersion, "2024-11-05");
  EXPECT_EQ(result->serverInfo.name, "lldb-mcp");
  EXPECT_EQ(result->serverInfo.version, "0.1.0");
  EXPECT_TRUE(result->capabilities.supportsToolsList);
  EXPECT_TRUE(result->capabilities.supportsResourcesList);
}

TEST_F(MCPClientTest, ToolsList) {
  llvm::Expected<ListToolsResult> result =
      Capture<ListToolsResult>([&](Client::Reply<ListToolsResult> reply) {
        client->ToolsList(std::move(reply));
      });
  ASSERT_THAT_EXPECTED(result, Succeeded());
  EXPECT_THAT(result->tools, testing::SizeIs(2));
}

TEST_F(MCPClientTest, ToolsCall) {
  llvm::Expected<CallToolResult> result =
      Capture<CallToolResult>([&](Client::Reply<CallToolResult> reply) {
        client->ToolsCall(CallToolParams{/*name=*/"echo", /*arguments=*/{}},
                          std::move(reply));
      });
  ASSERT_THAT_EXPECTED(result, Succeeded());
  ASSERT_THAT(result->content, testing::SizeIs(1));
  EXPECT_EQ(result->content.front().text, "echo");
  EXPECT_FALSE(result->isError);
}

TEST_F(MCPClientTest, ToolsCallError) {
  llvm::Expected<CallToolResult> result =
      Capture<CallToolResult>([&](Client::Reply<CallToolResult> reply) {
        client->ToolsCall(CallToolParams{/*name=*/"error", /*arguments=*/{}},
                          std::move(reply));
      });
  EXPECT_THAT_EXPECTED(result, FailedWithMessage("boom"));
}

TEST_F(MCPClientTest, ResourcesList) {
  llvm::Expected<ListResourcesResult> result = Capture<ListResourcesResult>(
      [&](Client::Reply<ListResourcesResult> reply) {
        client->ResourcesList(std::move(reply));
      });
  ASSERT_THAT_EXPECTED(result, Succeeded());
  ASSERT_THAT(result->resources, testing::SizeIs(1));
  EXPECT_EQ(result->resources.front().uri, "lldb://foo/bar");
}

TEST_F(MCPClientTest, ResourcesRead) {
  llvm::Expected<ReadResourceResult> result =
      Capture<ReadResourceResult>([&](Client::Reply<ReadResourceResult> reply) {
        client->ResourcesRead(ReadResourceParams{/*uri=*/"lldb://foo/bar"},
                              std::move(reply));
      });
  ASSERT_THAT_EXPECTED(result, Succeeded());
  ASSERT_THAT(result->contents, testing::SizeIs(1));
  EXPECT_EQ(result->contents.front().text, "foobar");
}

TEST_F(MCPClientTest, DisconnectHandler) {
  bool disconnected = false;
  client->SetDisconnectHandler([&]() { disconnected = true; });

  client_transport->SimulateClosed();
  EXPECT_TRUE(disconnected);
}

TEST_F(MCPClientTest, ErrorHandler) {
  std::string message;
  client->SetErrorHandler(
      [&](llvm::Error error) { message = llvm::toString(std::move(error)); });

  client_transport->SimulateError(llvm::createStringError("kaboom"));
  EXPECT_EQ(message, "kaboom");
}

#endif // ifndef _WIN32
