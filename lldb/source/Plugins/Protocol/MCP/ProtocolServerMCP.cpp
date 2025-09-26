//===- ProtocolServerMCP.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProtocolServerMCP.h"
#include "Resource.h"
#include "Tool.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Protocol/MCP/Server.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Threading.h"
#include <thread>

using namespace lldb_private;
using namespace lldb_private::mcp;
using namespace lldb_protocol::mcp;
using namespace llvm;

LLDB_PLUGIN_DEFINE(ProtocolServerMCP)

static constexpr llvm::StringLiteral kName = "lldb-mcp";
static constexpr llvm::StringLiteral kVersion = "0.1.0";

ProtocolServerMCP::ProtocolServerMCP() : ProtocolServer() {}

ProtocolServerMCP::~ProtocolServerMCP() { llvm::consumeError(Stop()); }

void ProtocolServerMCP::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void ProtocolServerMCP::Terminate() {
  if (llvm::Error error = ProtocolServer::Terminate())
    LLDB_LOG_ERROR(GetLog(LLDBLog::Host), std::move(error), "{0}");
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb::ProtocolServerUP ProtocolServerMCP::CreateInstance() {
  return std::make_unique<ProtocolServerMCP>();
}

llvm::StringRef ProtocolServerMCP::GetPluginDescriptionStatic() {
  return "MCP Server.";
}

void ProtocolServerMCP::Extend(lldb_protocol::mcp::Server &server) const {
  server.AddNotificationHandler("notifications/initialized",
                                [](const lldb_protocol::mcp::Notification &) {
                                  LLDB_LOG(GetLog(LLDBLog::Host),
                                           "MCP initialization complete");
                                });
  server.AddTool(
      std::make_unique<CommandTool>("command", "Run an lldb command."));
  server.AddTool(std::make_unique<DebuggerListTool>(
      "debugger_list", "List debugger instances with their debugger_id."));
  server.AddResourceProvider(std::make_unique<DebuggerResourceProvider>());
}

void ProtocolServerMCP::AcceptCallback(std::unique_ptr<Socket> socket) {
  Log *log = GetLog(LLDBLog::Host);
  std::string client_name = llvm::formatv("client_{0}", ++m_client_count);
  LLDB_LOG(log, "New MCP client connected: {0}", client_name);

  lldb::IOObjectSP io_sp = std::move(socket);
  auto transport_up = std::make_unique<lldb_protocol::mcp::Transport>(
      io_sp, io_sp, [client_name](llvm::StringRef message) {
        LLDB_LOG(GetLog(LLDBLog::Host), "{0}: {1}", client_name, message);
      });
  MCPTransport *transport_ptr = transport_up.get();
  auto instance_up = std::make_unique<lldb_protocol::mcp::Server>(
      std::string(kName), std::string(kVersion), *transport_up,
      /*log_callback=*/
      [client_name](llvm::StringRef message) {
        LLDB_LOG(GetLog(LLDBLog::Host), "{0} Server: {1}", client_name,
                 message);
      },
      /*closed_callback=*/
      [this, transport_ptr]() { m_instances.erase(transport_ptr); });
  Extend(*instance_up);
  llvm::Expected<MainLoop::ReadHandleUP> handle =
      transport_up->RegisterMessageHandler(m_loop, *instance_up);
  if (!handle) {
    LLDB_LOG_ERROR(log, handle.takeError(), "Failed to run MCP server: {0}");
    return;
  }
  m_instances[transport_ptr] =
      std::make_tuple<ServerUP, ReadHandleUP, TransportUP>(
          std::move(instance_up), std::move(*handle), std::move(transport_up));
}

llvm::Error ProtocolServerMCP::Start(ProtocolServer::Connection connection) {
  std::lock_guard<std::mutex> guard(m_mutex);

  if (m_running)
    return llvm::createStringError("the MCP server is already running");

  Status status;
  m_listener = Socket::Create(connection.protocol, status);
  if (status.Fail())
    return status.takeError();

  status = m_listener->Listen(connection.name, /*backlog=*/5);
  if (status.Fail())
    return status.takeError();

  auto handles =
      m_listener->Accept(m_loop, std::bind(&ProtocolServerMCP::AcceptCallback,
                                           this, std::placeholders::_1));
  if (llvm::Error error = handles.takeError())
    return error;

  auto listening_uris = m_listener->GetListeningConnectionURI();
  if (listening_uris.empty())
    return createStringError("failed to get listening connections");
  std::string address =
      llvm::join(m_listener->GetListeningConnectionURI(), ", ");

  ServerInfo info{listening_uris[0]};
  llvm::Expected<ServerInfoHandle> handle = ServerInfo::Write(info);
  if (!handle)
    return handle.takeError();

  m_running = true;
  m_server_info_handle = std::move(*handle);
  m_listen_handlers = std::move(*handles);
  m_loop_thread = std::thread([=] {
    llvm::set_thread_name("protocol-server.mcp");
    m_loop.Run();
  });

  return llvm::Error::success();
}

llvm::Error ProtocolServerMCP::Stop() {
  {
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!m_running)
      return createStringError("the MCP sever is not running");
    m_running = false;
  }

  // Stop the main loop.
  m_loop.AddPendingCallback(
      [](lldb_private::MainLoopBase &loop) { loop.RequestTermination(); });

  // Wait for the main loop to exit.
  if (m_loop_thread.joinable())
    m_loop_thread.join();

  m_server_info_handle.Remove();
  m_listen_handlers.clear();
  m_instances.clear();

  return llvm::Error::success();
}
