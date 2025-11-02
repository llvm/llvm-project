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
#include "lldb/Protocol/MCP/MCPError.h"
#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Transport.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Signals.h"

using namespace llvm;
using namespace lldb_private;
using namespace lldb_protocol::mcp;

ServerInfoHandle::ServerInfoHandle(StringRef filename) : m_filename(filename) {
  if (!m_filename.empty())
    sys::RemoveFileOnSignal(m_filename);
}

ServerInfoHandle::~ServerInfoHandle() { Remove(); }

ServerInfoHandle::ServerInfoHandle(ServerInfoHandle &&other) {
  *this = std::move(other);
}

ServerInfoHandle &
ServerInfoHandle::operator=(ServerInfoHandle &&other) noexcept {
  m_filename = std::move(other.m_filename);
  return *this;
}

void ServerInfoHandle::Remove() {
  if (m_filename.empty())
    return;

  sys::fs::remove(m_filename);
  sys::DontRemoveFileOnSignal(m_filename);
  m_filename.clear();
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

Server::Server(std::string name, std::string version, LogCallback log_callback)
    : m_name(std::move(name)), m_version(std::move(version)),
      m_log_callback(std::move(log_callback)) {}

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

MCPBinderUP Server::Bind(MCPTransport &transport) {
  MCPBinderUP binder_up = std::make_unique<MCPBinder>(transport);
  binder_up->Bind<InitializeResult, InitializeParams>(
      "initialize", &Server::InitializeHandler, this);
  binder_up->Bind<ListToolsResult, void>("tools/list",
                                         &Server::ToolsListHandler, this);
  binder_up->Bind<CallToolResult, CallToolParams>(
      "tools/call", &Server::ToolsCallHandler, this);
  binder_up->Bind<ListResourcesResult, void>(
      "resources/list", &Server::ResourcesListHandler, this);
  binder_up->Bind<ReadResourceResult, ReadResourceParams>(
      "resources/read", &Server::ResourcesReadHandler, this);
  binder_up->Bind<void>("notifications/initialized",
                        [this]() { Log("MCP initialization complete"); });
  return binder_up;
}

llvm::Error Server::Accept(MainLoop &loop, MCPTransportUP transport) {
  MCPBinderUP binder = Bind(*transport);
  MCPTransport *transport_ptr = transport.get();
  binder->OnDisconnect([this, transport_ptr]() {
    assert(m_instances.find(transport_ptr) != m_instances.end() &&
           "Client not found in m_instances");
    m_instances.erase(transport_ptr);
  });
  binder->OnError([this](llvm::Error err) {
    Logv("Transport error: {0}", llvm::toString(std::move(err)));
  });

  auto handle = transport->RegisterMessageHandler(loop, *binder);
  if (!handle)
    return handle.takeError();

  m_instances[transport_ptr] =
      Client{std::move(*handle), std::move(transport), std::move(binder)};
  return llvm::Error::success();
}

Expected<InitializeResult>
Server::InitializeHandler(const InitializeParams &request) {
  InitializeResult result;
  result.protocolVersion = mcp::kProtocolVersion;
  result.capabilities = GetCapabilities();
  result.serverInfo.name = m_name;
  result.serverInfo.version = m_version;
  return result;
}

llvm::Expected<ListToolsResult> Server::ToolsListHandler() {
  ListToolsResult result;
  for (const auto &tool : m_tools)
    result.tools.emplace_back(tool.second->GetDefinition());

  return result;
}

llvm::Expected<CallToolResult>
Server::ToolsCallHandler(const CallToolParams &params) {
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

  return text_result;
}

llvm::Expected<ListResourcesResult> Server::ResourcesListHandler() {
  ListResourcesResult result;
  for (std::unique_ptr<ResourceProvider> &resource_provider_up :
       m_resource_providers)
    for (const Resource &resource : resource_provider_up->GetResources())
      result.resources.push_back(resource);

  return result;
}

Expected<ReadResourceResult>
Server::ResourcesReadHandler(const ReadResourceParams &params) {
  StringRef uri_str = params.uri;
  if (uri_str.empty())
    return createStringError("no resource uri");

  for (std::unique_ptr<ResourceProvider> &resource_provider_up :
       m_resource_providers) {
    Expected<ReadResourceResult> result =
        resource_provider_up->ReadResource(uri_str);
    if (result.errorIsA<UnsupportedURI>()) {
      consumeError(result.takeError());
      continue;
    }
    if (!result)
      return result.takeError();

    return *result;
  }

  return make_error<MCPError>(
      formatv("no resource handler for uri: {0}", uri_str).str(),
      MCPError::kResourceNotFound);
}

ServerCapabilities Server::GetCapabilities() {
  lldb_protocol::mcp::ServerCapabilities capabilities;
  capabilities.supportsToolsList = true;
  capabilities.supportsResourcesList = true;
  // FIXME: Support sending notifications when a debugger/target are
  // added/removed.
  capabilities.supportsResourcesSubscribe = false;
  return capabilities;
}
