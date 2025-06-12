//===- ProtocolServerMCP.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProtocolServerMCP.h"
#include "MCPError.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Threading.h"
#include <thread>

using namespace lldb_private;
using namespace lldb_private::mcp;
using namespace llvm;

LLDB_PLUGIN_DEFINE(ProtocolServerMCP)

ProtocolServerMCP::ProtocolServerMCP(Debugger &debugger)
    : ProtocolServer(), m_debugger(debugger) {
  AddHandler("initialize", std::bind(&ProtocolServerMCP::InitializeHandler,
                                     this, std::placeholders::_1));
  AddHandler("tools/list", std::bind(&ProtocolServerMCP::ToolsListHandler, this,
                                     std::placeholders::_1));
  AddHandler("tools/call", std::bind(&ProtocolServerMCP::ToolsCallHandler, this,
                                     std::placeholders::_1));
  AddTool(std::make_unique<LLDBCommandTool>(
      "lldb_command", "Run an lldb command.", m_debugger));
}

ProtocolServerMCP::~ProtocolServerMCP() { llvm::consumeError(Stop()); }

void ProtocolServerMCP::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void ProtocolServerMCP::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb::ProtocolServerSP ProtocolServerMCP::CreateInstance(Debugger &debugger) {
  return std::make_shared<ProtocolServerMCP>(debugger);
}

llvm::StringRef ProtocolServerMCP::GetPluginDescriptionStatic() {
  return "MCP Server.";
}

llvm::Expected<protocol::Response>
ProtocolServerMCP::Handle(protocol::Request request) {
  auto it = m_handlers.find(request.method);
  if (it != m_handlers.end())
    return it->second(request);

  return make_error<MCPError>(
      llvm::formatv("no handler for request: {0}", request.method).str(), 1);
}

llvm::Error ProtocolServerMCP::Start(ProtocolServer::Connection connection) {
  std::lock_guard<std::mutex> guard(m_server_mutex);

  if (m_running)
    return llvm::createStringError("server already running");

  Status status;
  m_listener = Socket::Create(connection.protocol, status);
  if (status.Fail())
    return status.takeError();

  status = m_listener->Listen(connection.name, /*backlog=*/5);
  if (status.Fail())
    return status.takeError();

  std::string address =
      llvm::join(m_listener->GetListeningConnectionURI(), ", ");
  Log *log = GetLog(LLDBLog::Host);
  LLDB_LOG(log, "MCP server started with connection listeners: {0}", address);

  auto handles = m_listener->Accept(m_loop, [=](std::unique_ptr<Socket> sock) {
    std::lock_guard<std::mutex> guard(m_server_mutex);

    const std::string client_name =
        llvm::formatv("client-{0}", m_clients.size() + 1).str();
    LLDB_LOG(log, "client {0} connected", client_name);

    lldb::IOObjectSP io(std::move(sock));

    m_clients.emplace_back(io, [=]() {
      llvm::set_thread_name(client_name + "-runloop");
      if (auto Err = Run(std::make_unique<JSONRPCTransport>(io, io)))
        LLDB_LOG_ERROR(GetLog(LLDBLog::Host), std::move(Err), "MCP Error: {0}");
    });
  });
  if (llvm::Error error = handles.takeError())
    return error;

  m_read_handles = std::move(*handles);
  m_loop_thread = std::thread([=] {
    llvm::set_thread_name("mcp-runloop");
    m_loop.Run();
  });

  return llvm::Error::success();
}

llvm::Error ProtocolServerMCP::Stop() {
  {
    std::lock_guard<std::mutex> guard(m_server_mutex);
    m_running = false;
  }

  // Stop accepting new connections.
  m_loop.AddPendingCallback(
      [](MainLoopBase &loop) { loop.RequestTermination(); });

  // Wait for the main loop to exit.
  if (m_loop_thread.joinable())
    m_loop_thread.join();

  // Wait for all our clients to exit.
  for (auto &client : m_clients) {
    client.first->Close();
    if (client.second.joinable())
      client.second.join();
  }

  {
    std::lock_guard<std::mutex> guard(m_server_mutex);
    m_listener.reset();
    m_read_handles.clear();
    m_clients.clear();
  }

  return llvm::Error::success();
}

llvm::Error ProtocolServerMCP::Run(std::unique_ptr<JSONTransport> transport) {
  Log *log = GetLog(LLDBLog::Host);

  while (true) {
    llvm::Expected<protocol::Request> request =
        transport->Read<protocol::Request>(std::chrono::seconds(1));
    if (request.errorIsA<TransportEOFError>() ||
        request.errorIsA<TransportInvalidError>()) {
      consumeError(request.takeError());
      break;
    }

    if (request.errorIsA<TransportTimeoutError>()) {
      consumeError(request.takeError());
      continue;
    }

    if (llvm::Error err = request.takeError()) {
      LLDB_LOG_ERROR(log, std::move(err), "{0}");
      continue;
    }

    protocol::Response response;
    llvm::Expected<protocol::Response> maybe_response = Handle(*request);
    if (!maybe_response) {
      llvm::handleAllErrors(
          maybe_response.takeError(),
          [&](const MCPError &err) {
            response.error.emplace(err.toProtcolError());
          },
          [&](const llvm::ErrorInfoBase &err) {
            response.error.emplace();
            response.error->code = -1;
            response.error->message = err.message();
          });
    } else {
      response = *maybe_response;
    }

    response.id = request->id;

    if (llvm::Error err = transport->Write(response))
      return err;
  }

  return llvm::Error::success();
}

protocol::Capabilities ProtocolServerMCP::GetCapabilities() {
  protocol::Capabilities capabilities;
  capabilities.tools.listChanged = true;
  return capabilities;
}

void ProtocolServerMCP::AddTool(std::unique_ptr<Tool> tool) {
  std::lock_guard<std::mutex> guard(m_server_mutex);

  if (!tool)
    return;
  m_tools[tool->GetName()] = std::move(tool);
}

void ProtocolServerMCP::AddHandler(
    llvm::StringRef method,
    std::function<protocol::Response(const protocol::Request &)> handler) {
  std::lock_guard<std::mutex> guard(m_server_mutex);

  m_handlers[method] = std::move(handler);
}

protocol::Response
ProtocolServerMCP::InitializeHandler(const protocol::Request &request) {
  protocol::Response response;

  std::string protocol_version = protocol::kProtocolVersion.str();
  if (request.params) {
    if (const json::Object *param_obj = request.params->getAsObject()) {
      if (const json::Value *val = param_obj->get("protocolVersion")) {
        if (auto protocol_version_str = val->getAsString()) {
          protocol_version = *protocol_version_str;
        }
      }
    }
  }

  response.result.emplace(llvm::json::Object{
      {"protocolVersion", protocol_version},
      {"capabilities", GetCapabilities()},
      {"serverInfo",
       llvm::json::Object{{"name", kName}, {"version", kVersion}}}});
  return response;
}

protocol::Response
ProtocolServerMCP::ToolsListHandler(const protocol::Request &request) {
  protocol::Response response;

  llvm::json::Array tools;
  for (const auto &tool : m_tools)
    tools.emplace_back(toJSON(tool.second->GetDefinition()));

  response.result.emplace(llvm::json::Object{{"tools", std::move(tools)}});

  return response;
}

protocol::Response
ProtocolServerMCP::ToolsCallHandler(const protocol::Request &request) {
  protocol::Response response;

  if (!request.params)
    return response;

  const json::Object *param_obj = request.params->getAsObject();
  if (!param_obj)
    return response;

  const json::Value *name = param_obj->get("name");
  if (!name)
    return response;

  llvm::StringRef tool_name = name->getAsString().value_or("");
  if (tool_name.empty())
    return response;

  auto it = m_tools.find(tool_name);
  if (it == m_tools.end())
    return response;

  const json::Value *args = param_obj->get("arguments");
  if (!args)
    return response;

  protocol::TextResult text_result = it->second->Call(*args);
  response.result.emplace(toJSON(text_result));

  return response;
}
