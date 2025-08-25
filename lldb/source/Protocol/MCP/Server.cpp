//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Protocol/MCP/Server.h"
#include "lldb/Protocol/MCP/MCPError.h"

using namespace lldb_protocol::mcp;
using namespace llvm;

Server::Server(std::string name, std::string version,
               std::unique_ptr<MCPTransport> transport_up,
               lldb_private::MainLoop &loop)
    : m_name(std::move(name)), m_version(std::move(version)),
      m_transport_up(std::move(transport_up)), m_loop(loop) {
  AddRequestHandlers();
}

void Server::AddRequestHandlers() {
  AddRequestHandler("initialize", std::bind(&Server::InitializeHandler, this,
                                            std::placeholders::_1));
  AddRequestHandler("tools/list", std::bind(&Server::ToolsListHandler, this,
                                            std::placeholders::_1));
  AddRequestHandler("tools/call", std::bind(&Server::ToolsCallHandler, this,
                                            std::placeholders::_1));
  AddRequestHandler("resources/list", std::bind(&Server::ResourcesListHandler,
                                                this, std::placeholders::_1));
  AddRequestHandler("resources/read", std::bind(&Server::ResourcesReadHandler,
                                                this, std::placeholders::_1));
}

llvm::Expected<Response> Server::Handle(Request request) {
  auto it = m_request_handlers.find(request.method);
  if (it != m_request_handlers.end()) {
    llvm::Expected<Response> response = it->second(request);
    if (!response)
      return response;
    response->id = request.id;
    return *response;
  }

  return llvm::make_error<MCPError>(
      llvm::formatv("no handler for request: {0}", request.method).str());
}

void Server::Handle(Notification notification) {
  auto it = m_notification_handlers.find(notification.method);
  if (it != m_notification_handlers.end()) {
    it->second(notification);
    return;
  }
}

llvm::Expected<std::optional<Message>>
Server::HandleData(llvm::StringRef data) {
  auto message = llvm::json::parse<Message>(/*JSON=*/data);
  if (!message)
    return message.takeError();

  if (const Request *request = std::get_if<Request>(&(*message))) {
    llvm::Expected<Response> response = Handle(*request);

    // Handle failures by converting them into an Error message.
    if (!response) {
      Error protocol_error;
      llvm::handleAllErrors(
          response.takeError(),
          [&](const MCPError &err) { protocol_error = err.toProtocolError(); },
          [&](const llvm::ErrorInfoBase &err) {
            protocol_error.code = MCPError::kInternalError;
            protocol_error.message = err.message();
          });
      Response error_response;
      error_response.id = request->id;
      error_response.result = std::move(protocol_error);
      return error_response;
    }

    return *response;
  }

  if (const Notification *notification =
          std::get_if<Notification>(&(*message))) {
    Handle(*notification);
    return std::nullopt;
  }

  if (std::get_if<Response>(&(*message)))
    return llvm::createStringError("unexpected MCP message: response");

  llvm_unreachable("all message types handled");
}

void Server::AddTool(std::unique_ptr<Tool> tool) {
  std::lock_guard<std::mutex> guard(m_mutex);

  if (!tool)
    return;
  m_tools[tool->GetName()] = std::move(tool);
}

void Server::AddResourceProvider(
    std::unique_ptr<ResourceProvider> resource_provider) {
  std::lock_guard<std::mutex> guard(m_mutex);

  if (!resource_provider)
    return;
  m_resource_providers.push_back(std::move(resource_provider));
}

void Server::AddRequestHandler(llvm::StringRef method, RequestHandler handler) {
  std::lock_guard<std::mutex> guard(m_mutex);
  m_request_handlers[method] = std::move(handler);
}

void Server::AddNotificationHandler(llvm::StringRef method,
                                    NotificationHandler handler) {
  std::lock_guard<std::mutex> guard(m_mutex);
  m_notification_handlers[method] = std::move(handler);
}

llvm::Expected<Response> Server::InitializeHandler(const Request &request) {
  Response response;
  response.result = llvm::json::Object{
      {"protocolVersion", mcp::kProtocolVersion},
      {"capabilities", GetCapabilities()},
      {"serverInfo",
       llvm::json::Object{{"name", m_name}, {"version", m_version}}}};
  return response;
}

llvm::Expected<Response> Server::ToolsListHandler(const Request &request) {
  Response response;

  llvm::json::Array tools;
  for (const auto &tool : m_tools)
    tools.emplace_back(toJSON(tool.second->GetDefinition()));

  response.result = llvm::json::Object{{"tools", std::move(tools)}};

  return response;
}

llvm::Expected<Response> Server::ToolsCallHandler(const Request &request) {
  Response response;

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

  ToolArguments tool_args;
  if (const json::Value *args = param_obj->get("arguments"))
    tool_args = *args;

  llvm::Expected<TextResult> text_result = it->second->Call(tool_args);
  if (!text_result)
    return text_result.takeError();

  response.result = toJSON(*text_result);

  return response;
}

llvm::Expected<Response> Server::ResourcesListHandler(const Request &request) {
  Response response;

  llvm::json::Array resources;

  std::lock_guard<std::mutex> guard(m_mutex);
  for (std::unique_ptr<ResourceProvider> &resource_provider_up :
       m_resource_providers) {
    for (const Resource &resource : resource_provider_up->GetResources())
      resources.push_back(resource);
  }
  response.result = llvm::json::Object{{"resources", std::move(resources)}};

  return response;
}

llvm::Expected<Response> Server::ResourcesReadHandler(const Request &request) {
  Response response;

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

  std::lock_guard<std::mutex> guard(m_mutex);
  for (std::unique_ptr<ResourceProvider> &resource_provider_up :
       m_resource_providers) {
    llvm::Expected<ResourceResult> result =
        resource_provider_up->ReadResource(uri_str);
    if (result.errorIsA<UnsupportedURI>()) {
      llvm::consumeError(result.takeError());
      continue;
    }
    if (!result)
      return result.takeError();

    Response response;
    response.result = std::move(*result);
    return response;
  }

  return make_error<MCPError>(
      llvm::formatv("no resource handler for uri: {0}", uri_str).str(),
      MCPError::kResourceNotFound);
}

Capabilities Server::GetCapabilities() {
  lldb_protocol::mcp::Capabilities capabilities;
  capabilities.tools.listChanged = true;
  // FIXME: Support sending notifications when a debugger/target are
  // added/removed.
  capabilities.resources.listChanged = false;
  return capabilities;
}

llvm::Error Server::Run() {
  auto handle = m_transport_up->RegisterMessageHandler(m_loop, *this);
  if (!handle)
    return handle.takeError();

  lldb_private::Status status = m_loop.Run();
  if (status.Fail())
    return status.takeError();

  return llvm::Error::success();
}

void Server::Received(const Request &request) {
  auto SendResponse = [this](const Response &response) {
    if (llvm::Error error = m_transport_up->Send(response))
      m_transport_up->Log(llvm::toString(std::move(error)));
  };

  llvm::Expected<Response> response = Handle(request);
  if (response)
    return SendResponse(*response);

  lldb_protocol::mcp::Error protocol_error;
  llvm::handleAllErrors(
      response.takeError(),
      [&](const MCPError &err) { protocol_error = err.toProtocolError(); },
      [&](const llvm::ErrorInfoBase &err) {
        protocol_error.code = MCPError::kInternalError;
        protocol_error.message = err.message();
      });
  Response error_response;
  error_response.id = request.id;
  error_response.result = std::move(protocol_error);
  SendResponse(error_response);
}

void Server::Received(const Response &response) {
  m_transport_up->Log("unexpected MCP message: response");
}

void Server::Received(const Notification &notification) {
  Handle(notification);
}

void Server::OnError(llvm::Error error) {
  m_transport_up->Log(llvm::toString(std::move(error)));
  TerminateLoop();
}

void Server::OnClosed() {
  m_transport_up->Log("EOF");
  TerminateLoop();
}

void Server::TerminateLoop() {
  m_loop.AddPendingCallback(
      [](lldb_private::MainLoopBase &loop) { loop.RequestTermination(); });
}
