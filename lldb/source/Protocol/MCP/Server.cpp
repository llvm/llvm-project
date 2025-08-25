//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Protocol/MCP/Server.h"
#include "lldb/Host/Socket.h"
#include "lldb/Protocol/MCP/Binder.h"
#include "lldb/Protocol/MCP/MCPError.h"
#include "lldb/Protocol/MCP/Protocol.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Threading.h"
#include <future>
#include <memory>

using namespace lldb_private;
using namespace lldb_protocol::mcp;
using namespace llvm;

Server::Server(std::string name, std::string version,
               std::unique_ptr<MCPTransport> transport_up,
               lldb_private::MainLoop &loop)
    : m_name(std::move(name)), m_version(std::move(version)),
      m_transport_up(std::move(transport_up)), m_loop(loop),
      m_binder(m_transport_up.get()) {
  m_binder.request("initialize", &Server::InitializeHandler, this);
  m_binder.request("tools/list", &Server::ToolsListHandler, this);
  m_binder.request("tools/call", &Server::ToolsCallHandler, this);
  m_binder.request("resources/list", &Server::ResourcesListHandler, this);
  m_binder.request("resources/read", &Server::ResourcesReadHandler, this);
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

Expected<InitializeResult>
Server::InitializeHandler(const InitializeParams &request) {
  InitializeResult result;
  result.protocolVersion = mcp::kProtocolVersion;
  result.capabilities = GetCapabilities();
  result.serverInfo = Implementation{m_name, "", m_version};
  return result;
}

llvm::Expected<ToolsListResult> Server::ToolsListHandler(const Void &) {
  ToolsListResult result;

  for (const auto &tool : m_tools)
    result.tools.emplace_back(tool.second->GetDefinition());

  return result;
}

llvm::Expected<ToolsCallResult>
Server::ToolsCallHandler(const ToolsCallParams &params) {
  if (params.name.empty())
    return llvm::createStringError("no tool name");

  llvm::StringRef tool_name = params.name;
  if (tool_name.empty())
    return llvm::createStringError("no tool name");

  auto it = m_tools.find(tool_name);
  if (it == m_tools.end())
    return llvm::createStringError(llvm::formatv("no tool \"{0}\"", tool_name));

  ToolArguments tool_args;
  if (params.arguments)
    tool_args = *params.arguments;

  std::promise<llvm::Expected<ToolsCallResult>> result_promise;
  it->second->Call(tool_args,
                   [&result_promise](llvm::Expected<ToolsCallResult> result) {
                     result_promise.set_value(std::move(result));
                   });
  return result_promise.get_future().get();
}

llvm::Expected<ResourcesListResult> Server::ResourcesListHandler(const Void &) {
  ResourcesListResult result;

  std::lock_guard<std::mutex> guard(m_mutex);
  for (std::unique_ptr<ResourceProvider> &resource_provider_up :
       m_resource_providers)
    for (const Resource &resource : resource_provider_up->GetResources())
      result.resources.push_back(resource);

  return result;
}

llvm::Expected<ResourcesReadResult>
Server::ResourcesReadHandler(const ResourcesReadParams &params) {
  ResourcesReadResult result;

  llvm::StringRef uri_str = params.URI;
  if (uri_str.empty())
    return llvm::createStringError("no resource uri");

  std::lock_guard<std::mutex> guard(m_mutex);
  for (std::unique_ptr<ResourceProvider> &resource_provider_up :
       m_resource_providers) {
    llvm::Expected<ResourcesReadResult> result =
        resource_provider_up->ReadResource(uri_str);
    if (result.errorIsA<UnsupportedURI>()) {
      llvm::consumeError(result.takeError());
      continue;
    }
    if (!result)
      return result.takeError();
    return *result;
  }

  return make_error<MCPError>(
      llvm::formatv("no resource handler for uri: {0}", uri_str).str(),
      MCPError::kResourceNotFound);
}

ServerCapabilities Server::GetCapabilities() {
  ServerCapabilities capabilities;
  capabilities.supportsToolsList = true;
  capabilities.supportsResourcesList = true;
  // FIXME: Support sending notifications when a debugger/target are
  // added/removed.
  // capabilities.supportsResourcesSubscribe = true;
  return capabilities;
}

llvm::Error Server::Run() {
  auto handle = m_transport_up->RegisterMessageHandler(m_loop, m_binder);
  if (!handle)
    return handle.takeError();

  lldb_private::Status status = m_loop.Run();
  if (status.Fail())
    return status.takeError();

  return llvm::Error::success();
}

void Server::TerminateLoop() {
  m_loop.AddPendingCallback(
      [](lldb_private::MainLoopBase &loop) { loop.RequestTermination(); });
}
