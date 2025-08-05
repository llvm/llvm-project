//===-- ProtocolServerMCPTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "Plugins/Protocol/MCP/MCPError.h"
#include "Plugins/Protocol/MCP/ProtocolServerMCP.h"
#include "TestingSupport/Host/SocketTestUtilities.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Core/ProtocolServer.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/JSONTransport.h"
#include "lldb/Host/Socket.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::mcp::protocol;

namespace {
class TestProtocolServerMCP : public lldb_private::mcp::ProtocolServerMCP {
public:
  using ProtocolServerMCP::AddNotificationHandler;
  using ProtocolServerMCP::AddRequestHandler;
  using ProtocolServerMCP::AddResourceProvider;
  using ProtocolServerMCP::AddTool;
  using ProtocolServerMCP::GetSocket;
  using ProtocolServerMCP::ProtocolServerMCP;
};

class TestJSONTransport : public lldb_private::JSONRPCTransport {
public:
  using JSONRPCTransport::JSONRPCTransport;
  using JSONRPCTransport::ReadImpl;
  using JSONRPCTransport::WriteImpl;
};

/// Test tool that returns it argument as text.
class TestTool : public mcp::Tool {
public:
  using mcp::Tool::Tool;

  virtual llvm::Expected<mcp::protocol::TextResult>
  Call(const ToolArguments &args) override {
    std::string argument;
    if (const json::Object *args_obj =
            std::get<json::Value>(args).getAsObject()) {
      if (const json::Value *s = args_obj->get("arguments")) {
        argument = s->getAsString().value_or("");
      }
    }

    mcp::protocol::TextResult text_result;
    text_result.content.emplace_back(mcp::protocol::TextContent{{argument}});
    return text_result;
  }
};

class TestResourceProvider : public mcp::ResourceProvider {
  using mcp::ResourceProvider::ResourceProvider;

  virtual std::vector<Resource> GetResources() const override {
    std::vector<Resource> resources;

    Resource resource;
    resource.uri = "lldb://foo/bar";
    resource.name = "name";
    resource.description = "description";
    resource.mimeType = "application/json";

    resources.push_back(resource);
    return resources;
  }

  virtual llvm::Expected<ResourceResult>
  ReadResource(llvm::StringRef uri) const override {
    if (uri != "lldb://foo/bar")
      return llvm::make_error<mcp::UnsupportedURI>(uri.str());

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
class ErrorTool : public mcp::Tool {
public:
  using mcp::Tool::Tool;

  virtual llvm::Expected<mcp::protocol::TextResult>
  Call(const ToolArguments &args) override {
    return llvm::createStringError("error");
  }
};

/// Test tool that fails but doesn't return an error.
class FailTool : public mcp::Tool {
public:
  using mcp::Tool::Tool;

  virtual llvm::Expected<mcp::protocol::TextResult>
  Call(const ToolArguments &args) override {
    mcp::protocol::TextResult text_result;
    text_result.content.emplace_back(mcp::protocol::TextContent{{"failed"}});
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

  static constexpr llvm::StringLiteral k_localhost = "localhost";

  llvm::Error Write(llvm::StringRef message) {
    return m_transport_up->WriteImpl(llvm::formatv("{0}\n", message).str());
  }

  llvm::Expected<std::string> Read() {
    return m_transport_up->ReadImpl(std::chrono::milliseconds(100));
  }

  void SetUp() {
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
    m_server_up->AddTool(std::make_unique<TestTool>("test", "test tool"));
    m_server_up->AddResourceProvider(std::make_unique<TestResourceProvider>());
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

  void TearDown() {
    // Stop the server.
    ASSERT_THAT_ERROR(m_server_up->Stop(), llvm::Succeeded());
  }
};

} // namespace

TEST_F(ProtocolServerMCPTest, Intialization) {
  llvm::StringLiteral request =
      R"json({"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"lldb-unit","version":"0.1.0"}},"jsonrpc":"2.0","id":0})json";
  llvm::StringLiteral response =
      R"json( {"id":0,"jsonrpc":"2.0","result":{"capabilities":{"resources":{"listChanged":false,"subscribe":false},"tools":{"listChanged":true}},"protocolVersion":"2024-11-05","serverInfo":{"name":"lldb-mcp","version":"0.1.0"}}})json";

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());

  llvm::Expected<std::string> response_str = Read();
  ASSERT_THAT_EXPECTED(response_str, llvm::Succeeded());

  llvm::Expected<json::Value> response_json = json::parse(*response_str);
  ASSERT_THAT_EXPECTED(response_json, llvm::Succeeded());

  llvm::Expected<json::Value> expected_json = json::parse(response);
  ASSERT_THAT_EXPECTED(expected_json, llvm::Succeeded());

  EXPECT_EQ(*response_json, *expected_json);
}

TEST_F(ProtocolServerMCPTest, ToolsList) {
  llvm::StringLiteral request =
      R"json({"method":"tools/list","params":{},"jsonrpc":"2.0","id":1})json";
  llvm::StringLiteral response =
      R"json({"id":1,"jsonrpc":"2.0","result":{"tools":[{"description":"test tool","inputSchema":{"type":"object"},"name":"test"},{"description":"Run an lldb command.","inputSchema":{"properties":{"arguments":{"type":"string"},"debugger_id":{"type":"number"}},"required":["debugger_id"],"type":"object"},"name":"lldb_command"}]}})json";

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());

  llvm::Expected<std::string> response_str = Read();
  ASSERT_THAT_EXPECTED(response_str, llvm::Succeeded());

  llvm::Expected<json::Value> response_json = json::parse(*response_str);
  ASSERT_THAT_EXPECTED(response_json, llvm::Succeeded());

  llvm::Expected<json::Value> expected_json = json::parse(response);
  ASSERT_THAT_EXPECTED(expected_json, llvm::Succeeded());

  EXPECT_EQ(*response_json, *expected_json);
}

TEST_F(ProtocolServerMCPTest, ResourcesList) {
  llvm::StringLiteral request =
      R"json({"method":"resources/list","params":{},"jsonrpc":"2.0","id":2})json";
  llvm::StringLiteral response =
      R"json({"id":2,"jsonrpc":"2.0","result":{"resources":[{"description":"description","mimeType":"application/json","name":"name","uri":"lldb://foo/bar"}]}})json";

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());

  llvm::Expected<std::string> response_str = Read();
  ASSERT_THAT_EXPECTED(response_str, llvm::Succeeded());

  llvm::Expected<json::Value> response_json = json::parse(*response_str);
  ASSERT_THAT_EXPECTED(response_json, llvm::Succeeded());

  llvm::Expected<json::Value> expected_json = json::parse(response);
  ASSERT_THAT_EXPECTED(expected_json, llvm::Succeeded());

  EXPECT_EQ(*response_json, *expected_json);
}

TEST_F(ProtocolServerMCPTest, ToolsCall) {
  llvm::StringLiteral request =
      R"json({"method":"tools/call","params":{"name":"test","arguments":{"arguments":"foo","debugger_id":0}},"jsonrpc":"2.0","id":11})json";
  llvm::StringLiteral response =
      R"json({"id":11,"jsonrpc":"2.0","result":{"content":[{"text":"foo","type":"text"}],"isError":false}})json";

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());

  llvm::Expected<std::string> response_str = Read();
  ASSERT_THAT_EXPECTED(response_str, llvm::Succeeded());

  llvm::Expected<json::Value> response_json = json::parse(*response_str);
  ASSERT_THAT_EXPECTED(response_json, llvm::Succeeded());

  llvm::Expected<json::Value> expected_json = json::parse(response);
  ASSERT_THAT_EXPECTED(expected_json, llvm::Succeeded());

  EXPECT_EQ(*response_json, *expected_json);
}

TEST_F(ProtocolServerMCPTest, ToolsCallError) {
  m_server_up->AddTool(std::make_unique<ErrorTool>("error", "error tool"));

  llvm::StringLiteral request =
      R"json({"method":"tools/call","params":{"name":"error","arguments":{"arguments":"foo","debugger_id":0}},"jsonrpc":"2.0","id":11})json";
  llvm::StringLiteral response =
      R"json({"error":{"code":-32603,"message":"error"},"id":11,"jsonrpc":"2.0"})json";

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());

  llvm::Expected<std::string> response_str = Read();
  ASSERT_THAT_EXPECTED(response_str, llvm::Succeeded());

  llvm::Expected<json::Value> response_json = json::parse(*response_str);
  ASSERT_THAT_EXPECTED(response_json, llvm::Succeeded());

  llvm::Expected<json::Value> expected_json = json::parse(response);
  ASSERT_THAT_EXPECTED(expected_json, llvm::Succeeded());

  EXPECT_EQ(*response_json, *expected_json);
}

TEST_F(ProtocolServerMCPTest, ToolsCallFail) {
  m_server_up->AddTool(std::make_unique<FailTool>("fail", "fail tool"));

  llvm::StringLiteral request =
      R"json({"method":"tools/call","params":{"name":"fail","arguments":{"arguments":"foo","debugger_id":0}},"jsonrpc":"2.0","id":11})json";
  llvm::StringLiteral response =
      R"json({"id":11,"jsonrpc":"2.0","result":{"content":[{"text":"failed","type":"text"}],"isError":true}})json";

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());

  llvm::Expected<std::string> response_str = Read();
  ASSERT_THAT_EXPECTED(response_str, llvm::Succeeded());

  llvm::Expected<json::Value> response_json = json::parse(*response_str);
  ASSERT_THAT_EXPECTED(response_json, llvm::Succeeded());

  llvm::Expected<json::Value> expected_json = json::parse(response);
  ASSERT_THAT_EXPECTED(expected_json, llvm::Succeeded());

  EXPECT_EQ(*response_json, *expected_json);
}

TEST_F(ProtocolServerMCPTest, NotificationInitialized) {
  bool handler_called = false;
  std::condition_variable cv;
  std::mutex mutex;

  m_server_up->AddNotificationHandler(
      "notifications/initialized",
      [&](const mcp::protocol::Notification &notification) {
        {
          std::lock_guard<std::mutex> lock(mutex);
          handler_called = true;
        }
        cv.notify_all();
      });
  llvm::StringLiteral request =
      R"json({"method":"notifications/initialized","jsonrpc":"2.0"})json";

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());

  std::unique_lock<std::mutex> lock(mutex);
  cv.wait(lock, [&] { return handler_called; });
}
