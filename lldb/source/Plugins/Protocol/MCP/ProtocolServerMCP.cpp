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
#include <variant>

using namespace lldb_private;
using namespace lldb_private::mcp;
using namespace llvm;

LLDB_PLUGIN_DEFINE(ProtocolServerMCP)

static constexpr size_t kChunkSize = 1024;

ProtocolServerMCP::ProtocolServerMCP() : ProtocolServer() {
  AddRequestHandler("initialize",
                    std::bind(&ProtocolServerMCP::InitializeHandler, this,
                              std::placeholders::_1));

  AddRequestHandler("tools/list",
                    std::bind(&ProtocolServerMCP::ToolsListHandler, this,
                              std::placeholders::_1));
  AddRequestHandler("tools/call",
                    std::bind(&ProtocolServerMCP::ToolsCallHandler, this,
                              std::placeholders::_1));

  AddRequestHandler("resources/list",
                    std::bind(&ProtocolServerMCP::ResourcesListHandler, this,
                              std::placeholders::_1));
  AddRequestHandler("resources/read",
                    std::bind(&ProtocolServerMCP::ResourcesReadHandler, this,
                              std::placeholders::_1));
  AddNotificationHandler(
      "notifications/initialized", [](const protocol::Notification &) {
        LLDB_LOG(GetLog(LLDBLog::Host), "MCP initialization complete");
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

llvm::Expected<protocol::Response>
ProtocolServerMCP::Handle(protocol::Request request) {
  auto it = m_request_handlers.find(request.method);
  if (it != m_request_handlers.end()) {
    llvm::Expected<protocol::Response> response = it->second(request);
    if (!response)
      return response;
    response->id = request.id;
    return *response;
  }

  return make_error<MCPError>(
      llvm::formatv("no handler for request: {0}", request.method).str());
}

void ProtocolServerMCP::Handle(protocol::Notification notification) {
  auto it = m_notification_handlers.find(notification.method);
  if (it != m_notification_handlers.end()) {
    it->second(notification);
    return;
  }

  LLDB_LOG(GetLog(LLDBLog::Host), "MPC notification: {0} ({1})",
           notification.method, notification.params);
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
        if (Error error = ReadCallback(*client)) {
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
    llvm::Expected<std::optional<protocol::Message>> message =
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
  std::lock_guard<std::mutex> guard(m_server_mutex);

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
    std::lock_guard<std::mutex> guard(m_server_mutex);
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
    std::lock_guard<std::mutex> guard(m_server_mutex);
    m_listener.reset();
    m_listen_handlers.clear();
    m_clients.clear();
  }

  return llvm::Error::success();
}

llvm::Expected<std::optional<protocol::Message>>
ProtocolServerMCP::HandleData(llvm::StringRef data) {
  auto message = llvm::json::parse<protocol::Message>(/*JSON=*/data);
  if (!message)
    return message.takeError();

  if (const protocol::Request *request =
          std::get_if<protocol::Request>(&(*message))) {
    llvm::Expected<protocol::Response> response = Handle(*request);

    // Handle failures by converting them into an Error message.
    if (!response) {
      protocol::Error protocol_error;
      llvm::handleAllErrors(
          response.takeError(),
          [&](const MCPError &err) { protocol_error = err.toProtcolError(); },
          [&](const llvm::ErrorInfoBase &err) {
            protocol_error.error.code = MCPError::kInternalError;
            protocol_error.error.message = err.message();
          });
      protocol_error.id = request->id;
      return protocol_error;
    }

    return *response;
  }

  if (const protocol::Notification *notification =
          std::get_if<protocol::Notification>(&(*message))) {
    Handle(*notification);
    return std::nullopt;
  }

  if (std::get_if<protocol::Error>(&(*message)))
    return llvm::createStringError("unexpected MCP message: error");

  if (std::get_if<protocol::Response>(&(*message)))
    return llvm::createStringError("unexpected MCP message: response");

  llvm_unreachable("all message types handled");
}

protocol::Capabilities ProtocolServerMCP::GetCapabilities() {
  protocol::Capabilities capabilities;
  capabilities.tools.listChanged = true;
  // FIXME: Support sending notifications when a debugger/target are
  // added/removed.
  capabilities.resources.listChanged = false;
  return capabilities;
}

void ProtocolServerMCP::AddTool(std::unique_ptr<Tool> tool) {
  std::lock_guard<std::mutex> guard(m_server_mutex);

  if (!tool)
    return;
  m_tools[tool->GetName()] = std::move(tool);
}

void ProtocolServerMCP::AddResourceProvider(
    std::unique_ptr<ResourceProvider> resource_provider) {
  std::lock_guard<std::mutex> guard(m_server_mutex);

  if (!resource_provider)
    return;
  m_resource_providers.push_back(std::move(resource_provider));
}

void ProtocolServerMCP::AddRequestHandler(llvm::StringRef method,
                                          RequestHandler handler) {
  std::lock_guard<std::mutex> guard(m_server_mutex);
  m_request_handlers[method] = std::move(handler);
}

void ProtocolServerMCP::AddNotificationHandler(llvm::StringRef method,
                                               NotificationHandler handler) {
  std::lock_guard<std::mutex> guard(m_server_mutex);
  m_notification_handlers[method] = std::move(handler);
}

llvm::Expected<protocol::Response>
ProtocolServerMCP::InitializeHandler(const protocol::Request &request) {
  protocol::Response response;
  response.result.emplace(llvm::json::Object{
      {"protocolVersion", protocol::kVersion},
      {"capabilities", GetCapabilities()},
      {"serverInfo",
       llvm::json::Object{{"name", kName}, {"version", kVersion}}}});
  return response;
}

llvm::Expected<protocol::Response>
ProtocolServerMCP::ToolsListHandler(const protocol::Request &request) {
  protocol::Response response;

  llvm::json::Array tools;
  for (const auto &tool : m_tools)
    tools.emplace_back(toJSON(tool.second->GetDefinition()));

  response.result.emplace(llvm::json::Object{{"tools", std::move(tools)}});

  return response;
}

llvm::Expected<protocol::Response>
ProtocolServerMCP::ToolsCallHandler(const protocol::Request &request) {
  protocol::Response response;

  if (!request.params)
    return llvm::createStringError("no tool parameters");

  const json::Object *param_obj = request.params->getAsObject();
  if (!param_obj)
    return llvm::createStringError("no tool parameters");

  const json::Value *name = param_obj->get("name");
  if (!name)
    return llvm::createStringError("no tool name");

  llvm::StringRef tool_name = name->getAsString().value_or("");
  if (tool_name.empty())
    return llvm::createStringError("no tool name");

  auto it = m_tools.find(tool_name);
  if (it == m_tools.end())
    return llvm::createStringError(llvm::formatv("no tool \"{0}\"", tool_name));

  protocol::ToolArguments tool_args;
  if (const json::Value *args = param_obj->get("arguments"))
    tool_args = *args;

  llvm::Expected<protocol::TextResult> text_result =
      it->second->Call(tool_args);
  if (!text_result)
    return text_result.takeError();

  response.result.emplace(toJSON(*text_result));

  return response;
}

llvm::Expected<protocol::Response>
ProtocolServerMCP::ResourcesListHandler(const protocol::Request &request) {
  protocol::Response response;

  llvm::json::Array resources;

  std::lock_guard<std::mutex> guard(m_server_mutex);
  for (std::unique_ptr<ResourceProvider> &resource_provider_up :
       m_resource_providers) {
    for (const protocol::Resource &resource :
         resource_provider_up->GetResources())
      resources.push_back(resource);
  }
  response.result.emplace(
      llvm::json::Object{{"resources", std::move(resources)}});

  return response;
}

llvm::Expected<protocol::Response>
ProtocolServerMCP::ResourcesReadHandler(const protocol::Request &request) {
  protocol::Response response;

  if (!request.params)
    return llvm::createStringError("no resource parameters");

  const json::Object *param_obj = request.params->getAsObject();
  if (!param_obj)
    return llvm::createStringError("no resource parameters");

  const json::Value *uri = param_obj->get("uri");
  if (!uri)
    return llvm::createStringError("no resource uri");

  llvm::StringRef uri_str = uri->getAsString().value_or("");
  if (uri_str.empty())
    return llvm::createStringError("no resource uri");

  std::lock_guard<std::mutex> guard(m_server_mutex);
  for (std::unique_ptr<ResourceProvider> &resource_provider_up :
       m_resource_providers) {
    llvm::Expected<protocol::ResourceResult> result =
        resource_provider_up->ReadResource(uri_str);
    if (result.errorIsA<UnsupportedURI>()) {
      llvm::consumeError(result.takeError());
      continue;
    }
    if (!result)
      return result.takeError();

    protocol::Response response;
    response.result.emplace(std::move(*result));
    return response;
  }

  return make_error<MCPError>(
      llvm::formatv("no resource handler for uri: {0}", uri_str).str(),
      MCPError::kResourceNotFound);
}
