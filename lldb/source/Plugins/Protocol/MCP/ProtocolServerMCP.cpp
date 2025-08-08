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
#include "lldb/Protocol/MCP/MCPError.h"
#include "lldb/Protocol/MCP/Tool.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Threading.h"
#include <thread>
#include <variant>

using namespace lldb_private;
using namespace lldb_private::mcp;
using namespace lldb_protocol::mcp;
using namespace llvm;

LLDB_PLUGIN_DEFINE(ProtocolServerMCP)

static constexpr size_t kChunkSize = 1024;
static constexpr llvm::StringLiteral kName = "lldb-mcp";
static constexpr llvm::StringLiteral kVersion = "0.1.0";

ProtocolServerMCP::ProtocolServerMCP()
    : ProtocolServer(),
      lldb_protocol::mcp::Server(std::string(kName), std::string(kVersion)) {
  AddNotificationHandler("notifications/initialized",
                         [](const lldb_protocol::mcp::Notification &) {
                           LLDB_LOG(GetLog(LLDBLog::Host),
                                    "MCP initialization complete");
                         });

  AddTool(
      std::make_unique<CommandTool>("lldb_command", "Run an lldb command."));

  AddResourceProvider(std::make_unique<DebuggerResourceProvider>());
}

ProtocolServerMCP::~ProtocolServerMCP() { llvm::consumeError(Stop()); }

void ProtocolServerMCP::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void ProtocolServerMCP::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb::ProtocolServerUP ProtocolServerMCP::CreateInstance() {
  return std::make_unique<ProtocolServerMCP>();
}

llvm::StringRef ProtocolServerMCP::GetPluginDescriptionStatic() {
  return "MCP Server.";
}

void ProtocolServerMCP::AcceptCallback(std::unique_ptr<Socket> socket) {
  LLDB_LOG(GetLog(LLDBLog::Host), "New MCP client ({0}) connected",
           m_clients.size() + 1);

  lldb::IOObjectSP io_sp = std::move(socket);
  auto client_up = std::make_unique<Client>();
  client_up->io_sp = io_sp;
  Client *client = client_up.get();

  Status status;
  auto read_handle_up = m_loop.RegisterReadObject(
      io_sp,
      [this, client](MainLoopBase &loop) {
        if (llvm::Error error = ReadCallback(*client)) {
          LLDB_LOG_ERROR(GetLog(LLDBLog::Host), std::move(error), "{0}");
          client->read_handle_up.reset();
        }
      },
      status);
  if (status.Fail())
    return;

  client_up->read_handle_up = std::move(read_handle_up);
  m_clients.emplace_back(std::move(client_up));
}

llvm::Error ProtocolServerMCP::ReadCallback(Client &client) {
  char chunk[kChunkSize];
  size_t bytes_read = sizeof(chunk);
  if (Status status = client.io_sp->Read(chunk, bytes_read); status.Fail())
    return status.takeError();
  client.buffer.append(chunk, bytes_read);

  for (std::string::size_type pos;
       (pos = client.buffer.find('\n')) != std::string::npos;) {
    llvm::Expected<std::optional<lldb_protocol::mcp::Message>> message =
        HandleData(StringRef(client.buffer.data(), pos));
    client.buffer = client.buffer.erase(0, pos + 1);
    if (!message)
      return message.takeError();

    if (*message) {
      std::string Output;
      llvm::raw_string_ostream OS(Output);
      OS << llvm::formatv("{0}", toJSON(**message)) << '\n';
      size_t num_bytes = Output.size();
      return client.io_sp->Write(Output.data(), num_bytes).takeError();
    }
  }

  return llvm::Error::success();
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

  m_running = true;
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
      [](MainLoopBase &loop) { loop.RequestTermination(); });

  // Wait for the main loop to exit.
  if (m_loop_thread.joinable())
    m_loop_thread.join();

  {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_listener.reset();
    m_listen_handlers.clear();
    m_clients.clear();
  }

  return llvm::Error::success();
}

lldb_protocol::mcp::Capabilities ProtocolServerMCP::GetCapabilities() {
  lldb_protocol::mcp::Capabilities capabilities;
  capabilities.tools.listChanged = true;
  // FIXME: Support sending notifications when a debugger/target are
  // added/removed.
  capabilities.resources.listChanged = false;
  return capabilities;
}
