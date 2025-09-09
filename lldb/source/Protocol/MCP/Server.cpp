//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Protocol/MCP/Server.h"
#include "lldb/Host/File.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/JSONTransport.h"
#include "lldb/Protocol/MCP/MCPError.h"
#include "lldb/Protocol/MCP/Protocol.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Signals.h"

using namespace llvm;
using namespace lldb_private;
using namespace lldb_protocol::mcp;

ServerInfoHandle::ServerInfoHandle() : ServerInfoHandle("") {}

ServerInfoHandle::ServerInfoHandle(StringRef filename) : m_filename(filename) {
  if (!m_filename.empty())
    sys::RemoveFileOnSignal(m_filename);
}

ServerInfoHandle::~ServerInfoHandle() {
  if (m_filename.empty())
    return;

  sys::fs::remove(m_filename);
  sys::DontRemoveFileOnSignal(m_filename);
  m_filename.clear();
}

ServerInfoHandle::ServerInfoHandle(ServerInfoHandle &&other)
    : m_filename(other.m_filename) {
  *this = std::move(other);
}

ServerInfoHandle &
ServerInfoHandle::operator=(ServerInfoHandle &&other) noexcept {
  m_filename = other.m_filename;
  other.m_filename.clear();
  return *this;
}

json::Value lldb_protocol::mcp::toJSON(const ServerInfo &SM) {
  return json::Object{{"connection_uri", SM.connection_uri}};
}

bool lldb_protocol::mcp::fromJSON(const json::Value &V, ServerInfo &SM,
                                  json::Path P) {
  json::ObjectMapper O(V, P);
  return O && O.map("connection_uri", SM.connection_uri);
}

Expected<ServerInfoHandle> ServerInfo::Write(const ServerInfo &info) {
  std::string buf = formatv("{0}", toJSON(info)).str();
  size_t num_bytes = buf.size();

  FileSpec user_lldb_dir = HostInfo::GetUserLLDBDir();

  Status error(sys::fs::create_directory(user_lldb_dir.GetPath()));
  if (error.Fail())
    return error.takeError();

  FileSpec mcp_registry_entry_path = user_lldb_dir.CopyByAppendingPathComponent(
      formatv("lldb-mcp-{0}.json", getpid()).str());

  const File::OpenOptions flags = File::eOpenOptionWriteOnly |
                                  File::eOpenOptionCanCreate |
                                  File::eOpenOptionTruncate;
  Expected<lldb::FileUP> file =
      FileSystem::Instance().Open(mcp_registry_entry_path, flags);
  if (!file)
    return file.takeError();
  if (llvm::Error error = (*file)->Write(buf.data(), num_bytes).takeError())
    return error;
  return ServerInfoHandle{mcp_registry_entry_path.GetPath()};
}

Expected<std::vector<ServerInfo>> ServerInfo::Load() {
  namespace path = llvm::sys::path;
  FileSpec user_lldb_dir = HostInfo::GetUserLLDBDir();
  FileSystem &fs = FileSystem::Instance();
  std::error_code EC;
  vfs::directory_iterator it = fs.DirBegin(user_lldb_dir, EC);
  vfs::directory_iterator end;
  std::vector<ServerInfo> infos;
  for (; it != end && !EC; it.increment(EC)) {
    auto &entry = *it;
    auto path = entry.path();
    auto name = path::filename(path);
    if (!name.starts_with("lldb-mcp-") || !name.ends_with(".json"))
      continue;

    auto buffer = fs.CreateDataBuffer(path);
    auto info = json::parse<ServerInfo>(toStringRef(buffer->GetData()));
    if (!info)
      return info.takeError();

    infos.emplace_back(std::move(*info));
  }

  return infos;
}

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

llvm::Expected<Response> Server::Handle(const Request &request) {
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

void Server::Handle(const Notification &notification) {
  auto it = m_notification_handlers.find(notification.method);
  if (it != m_notification_handlers.end()) {
    it->second(notification);
    return;
  }
}

void Server::AddTool(std::unique_ptr<Tool> tool) {
  if (!tool)
    return;
  m_tools[tool->GetName()] = std::move(tool);
}

void Server::AddResourceProvider(
    std::unique_ptr<ResourceProvider> resource_provider) {
  if (!resource_provider)
    return;
  m_resource_providers.push_back(std::move(resource_provider));
}

void Server::AddRequestHandler(llvm::StringRef method, RequestHandler handler) {
  m_request_handlers[method] = std::move(handler);
}

void Server::AddNotificationHandler(llvm::StringRef method,
                                    NotificationHandler handler) {
  m_notification_handlers[method] = std::move(handler);
}

llvm::Expected<Response> Server::InitializeHandler(const Request &request) {
  Response response;
  InitializeResult result;
  result.protocolVersion = mcp::kProtocolVersion;
  result.capabilities = GetCapabilities();
  result.serverInfo.name = m_name;
  result.serverInfo.version = m_version;
  response.result = std::move(result);
  return response;
}

llvm::Expected<Response> Server::ToolsListHandler(const Request &request) {
  Response response;

  ListToolsResult result;
  for (const auto &tool : m_tools)
    result.tools.emplace_back(tool.second->GetDefinition());

  response.result = std::move(result);

  return response;
}

llvm::Expected<Response> Server::ToolsCallHandler(const Request &request) {
  Response response;

  if (!request.params)
    return llvm::createStringError("no tool parameters");
  CallToolParams params;
  json::Path::Root root("params");
  if (!fromJSON(request.params, params, root))
    return root.getError();

  llvm::StringRef tool_name = params.name;
  if (tool_name.empty())
    return llvm::createStringError("no tool name");

  auto it = m_tools.find(tool_name);
  if (it == m_tools.end())
    return llvm::createStringError(llvm::formatv("no tool \"{0}\"", tool_name));

  ToolArguments tool_args;
  if (params.arguments)
    tool_args = *params.arguments;

  llvm::Expected<CallToolResult> text_result = it->second->Call(tool_args);
  if (!text_result)
    return text_result.takeError();

  response.result = toJSON(*text_result);

  return response;
}

llvm::Expected<Response> Server::ResourcesListHandler(const Request &request) {
  Response response;

  ListResourcesResult result;
  for (std::unique_ptr<ResourceProvider> &resource_provider_up :
       m_resource_providers)
    for (const Resource &resource : resource_provider_up->GetResources())
      result.resources.push_back(resource);

  response.result = std::move(result);

  return response;
}

llvm::Expected<Response> Server::ResourcesReadHandler(const Request &request) {
  Response response;

  if (!request.params)
    return llvm::createStringError("no resource parameters");

  ReadResourceParams params;
  json::Path::Root root("params");
  if (!fromJSON(request.params, params, root))
    return root.getError();

  llvm::StringRef uri_str = params.uri;
  if (uri_str.empty())
    return llvm::createStringError("no resource uri");

  for (std::unique_ptr<ResourceProvider> &resource_provider_up :
       m_resource_providers) {
    llvm::Expected<ReadResourceResult> result =
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

ServerCapabilities Server::GetCapabilities() {
  lldb_protocol::mcp::ServerCapabilities capabilities;
  capabilities.supportsToolsList = true;
  // FIXME: Support sending notifications when a debugger/target are
  // added/removed.
  capabilities.supportsResourcesList = false;
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
