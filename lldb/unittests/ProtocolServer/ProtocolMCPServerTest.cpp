//===-- ProtocolServerMCPTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "Plugins/Protocol/MCP/ProtocolServerMCP.h"
#include "TestingSupport/Host/JSONTransportTestUtilities.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/ProtocolServer.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/JSONTransport.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/MainLoopBase.h"
#include "lldb/Host/Socket.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Protocol/MCP/MCPError.h"
#include "lldb/Protocol/MCP/Protocol.h"
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
class TestProtocolServerMCP : public lldb_private::mcp::ProtocolServerMCP {
public:
  using ProtocolServerMCP::GetSocket;
  using ProtocolServerMCP::ProtocolServerMCP;

  using ExtendCallback =
      std::function<void(lldb_protocol::mcp::Server &server)>;

  virtual void Extend(lldb_protocol::mcp::Server &server) const override {
    if (m_extend_callback)
      m_extend_callback(server);
  };

  void Extend(ExtendCallback callback) { m_extend_callback = callback; }

  ExtendCallback m_extend_callback;
};

using Message = typename Transport<Request, Response, Notification>::Message;

class TestJSONTransport final
    : public lldb_private::JSONRPCTransport<Request, Response, Notification> {
public:
  using JSONRPCTransport::JSONRPCTransport;

  void Log(llvm::StringRef message) override {
    log_messages.emplace_back(message);
  }

  std::vector<std::string> log_messages;
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

class ProtocolServerMCPTest : public ::testing::Test {
public:
  SubsystemRAII<FileSystem, HostInfo, PlatformRemoteMacOSX, Socket> subsystems;
  DebuggerSP m_debugger_sp;

  lldb::IOObjectSP m_io_sp;
  std::unique_ptr<TestJSONTransport> m_transport_up;
  std::unique_ptr<TestProtocolServerMCP> m_server_up;
  MainLoop loop;
  MockMessageHandler<Request, Response, Notification> message_handler;

  static constexpr llvm::StringLiteral k_localhost = "localhost";

  llvm::Error Write(llvm::StringRef message) {
    std::string output = llvm::formatv("{0}\n", message).str();
    size_t bytes_written = output.size();
    return m_io_sp->Write(output.data(), bytes_written).takeError();
  }

  void CloseInput() {
    EXPECT_THAT_ERROR(m_io_sp->Close().takeError(), Succeeded());
  }

  /// Run the transport MainLoop and return any messages received.
  llvm::Error
  Run(std::chrono::milliseconds timeout = std::chrono::milliseconds(200)) {
    loop.AddCallback([](MainLoopBase &loop) { loop.RequestTermination(); },
                     timeout);
    auto handle = m_transport_up->RegisterMessageHandler(loop, message_handler);
    if (!handle)
      return handle.takeError();

    return loop.Run().takeError();
  }

  void SetUp() override {
    // Create a debugger.
    ArchSpec arch("arm64-apple-macosx-");
    Platform::SetHostPlatform(
        PlatformRemoteMacOSX::CreateInstance(true, &arch));
    m_debugger_sp = Debugger::CreateInstance();

    // Create & start the server.
    ProtocolServer::Connection connection;
    connection.protocol = Socket::SocketProtocol::ProtocolTcp;
    connection.name = llvm::formatv("{0}:0", k_localhost).str();
    m_server_up = std::make_unique<TestProtocolServerMCP>();
    m_server_up->Extend([&](auto &server) {
      server.AddTool(std::make_unique<TestTool>("test", "test tool"));
      server.AddResourceProvider(std::make_unique<TestResourceProvider>());
    });
    ASSERT_THAT_ERROR(m_server_up->Start(connection), llvm::Succeeded());

    // Connect to the server over a TCP socket.
    auto connect_socket_up = std::make_unique<TCPSocket>(true);
    ASSERT_THAT_ERROR(connect_socket_up
                          ->Connect(llvm::formatv("{0}:{1}", k_localhost,
                                                  static_cast<TCPSocket *>(
                                                      m_server_up->GetSocket())
                                                      ->GetLocalPortNumber())
                                        .str())
                          .ToError(),
                      llvm::Succeeded());

    // Set up JSON transport for the client.
    m_io_sp = std::move(connect_socket_up);
    m_transport_up = std::make_unique<TestJSONTransport>(m_io_sp, m_io_sp);
  }

  void TearDown() override {
    // Stop the server.
    ASSERT_THAT_ERROR(m_server_up->Stop(), llvm::Succeeded());
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
  m_server_up->Extend([&](auto &server) {
    server.AddTool(std::make_unique<ErrorTool>("error", "error tool"));
  });

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
  m_server_up->Extend([&](auto &server) {
    server.AddTool(std::make_unique<FailTool>("fail", "fail tool"));
  });

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
  std::mutex mutex;

  m_server_up->Extend([&](auto &server) {
    server.AddNotificationHandler("notifications/initialized",
                                  [&](const Notification &notification) {
                                    {
                                      std::lock_guard<std::mutex> lock(mutex);
                                      handler_called = true;
                                    }
                                    cv.notify_all();
                                  });
  });
  llvm::StringLiteral request =
      R"json({"method":"notifications/initialized","jsonrpc":"2.0"})json";

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());

  std::unique_lock<std::mutex> lock(mutex);
  cv.wait(lock, [&] { return handler_called; });
}
