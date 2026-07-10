//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Multiplexer.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace lldb_protocol::mcp;
using namespace lldb_mcp;

namespace {

/// Server name reported to the client during initialization.
constexpr llvm::StringLiteral kServerName = "lldb-mcp";

/// Client-facing tool names.
/// @{
constexpr llvm::StringLiteral kToolCommand = "command";
constexpr llvm::StringLiteral kToolSessionsList = "sessions_list";
/// @}

/// Backend tool names, as exposed by the LLDB MCP server.
/// @{
constexpr llvm::StringLiteral kBackendToolCommand = "command";
constexpr llvm::StringLiteral kBackendToolDebuggerList = "debugger_list";
/// @}

} // namespace

Multiplexer::Multiplexer(std::unique_ptr<MCPTransport> client_transport,
                         LogCallback log_callback)
    : m_client_transport(std::move(client_transport)),
      m_client_binder(std::make_unique<MCPBinder>(*m_client_transport)),
      m_log_callback(std::move(log_callback)) {
  m_client_binder->Bind<InitializeResult, InitializeParams>(
      "initialize", &Multiplexer::HandleInitialize, this);
  m_client_binder->Bind<ListToolsResult, void>(
      "tools/list", &Multiplexer::HandleToolsList, this);
  m_client_binder->BindAsync<CallToolResult, CallToolParams>(
      "tools/call", &Multiplexer::HandleToolsCall, this);
  m_client_binder->BindAsync<ListResourcesResult, void>(
      "resources/list", &Multiplexer::HandleResourcesList, this);
  m_client_binder->BindAsync<ReadResourceResult, ReadResourceParams>(
      "resources/read", &Multiplexer::HandleResourcesRead, this);
  m_client_binder->Bind<void>("notifications/initialized", [this]() {
    Log("client initialization complete");
  });
}

void Multiplexer::AddBackend(std::unique_ptr<Client> backend) {
  m_backends.push_back(std::move(backend));
}

llvm::Error Multiplexer::Run() {
  // Run is called only after a backend has been added. Starting with none is a
  // setup bug, not a usable server.
  if (m_backends.empty())
    return llvm::createStringError(
        "no backends registered before Multiplexer::Run");

  m_client_binder->OnDisconnect(&Multiplexer::HandleDisconnect, this);
  m_client_binder->OnError([this](llvm::Error err) {
    Log(formatv("client transport error: {0}", toString(std::move(err))).str());
  });
  return m_client_transport->RegisterMessageHandler(*m_client_binder);
}

void Multiplexer::SetDisconnectHandler(llvm::unique_function<void()> handler) {
  m_disconnect_handler = std::move(handler);
}

Client *Multiplexer::Route() {
  if (m_backends.empty())
    return nullptr;
  return m_backends.front().get();
}

Expected<InitializeResult>
Multiplexer::HandleInitialize(const InitializeParams &) {
  InitializeResult result;
  result.protocolVersion = kProtocolVersion;
  result.capabilities.supportsToolsList = true;
  result.capabilities.supportsResourcesList = true;
  result.serverInfo.name = kServerName;
  result.serverInfo.version = GetServerVersion();
  return result;
}

Expected<ListToolsResult> Multiplexer::HandleToolsList() {
  ListToolsResult result;

  ToolDefinition command;
  command.name = kToolCommand;
  command.description = "Run an LLDB command in a debug session.";
  command.inputSchema = json::Object{
      {"type", "object"},
      {"properties",
       json::Object{
           {"command",
            json::Object{{"type", "string"},
                         {"description", "The LLDB command to run."}}},
           {"debugger",
            json::Object{{"type", "string"},
                         {"description",
                          "The debugger id or URI selecting the debug "
                          "session. Defaults to the first."}}},
       }},
      {"required", json::Array{"command"}},
  };
  result.tools.push_back(std::move(command));

  ToolDefinition sessions_list;
  sessions_list.name = kToolSessionsList;
  sessions_list.description = "List the active debug sessions.";
  sessions_list.inputSchema = json::Object{{"type", "object"}};
  result.tools.push_back(std::move(sessions_list));

  return result;
}

void Multiplexer::HandleToolsCall(const CallToolParams &params,
                                  Reply<CallToolResult> reply) {
  Client *backend = Route();
  if (!backend)
    return reply(createStringError("no debug session available"));

  std::optional<StringRef> backend_tool =
      StringSwitch<std::optional<StringRef>>(params.name)
          .Case(kToolCommand, kBackendToolCommand)
          .Case(kToolSessionsList, kBackendToolDebuggerList)
          .Default(std::nullopt);
  if (!backend_tool)
    return reply(createStringError(formatv("no tool \"{0}\"", params.name)));

  CallToolParams backend_params;
  backend_params.arguments = params.arguments;
  backend_params.name = *backend_tool;

  backend->ToolsCall(backend_params, std::move(reply));
}

void Multiplexer::HandleResourcesList(Reply<ListResourcesResult> reply) {
  Client *backend = Route();
  if (!backend)
    return reply(ListResourcesResult{});
  backend->ResourcesList(std::move(reply));
}

void Multiplexer::HandleResourcesRead(const ReadResourceParams &params,
                                      Reply<ReadResourceResult> reply) {
  Client *backend = Route();
  if (!backend)
    return reply(createStringError("no debug session available"));
  backend->ResourcesRead(params, std::move(reply));
}

void Multiplexer::HandleDisconnect() {
  if (m_disconnect_handler)
    m_disconnect_handler();
}

void Multiplexer::Log(llvm::StringRef message) {
  if (m_log_callback)
    m_log_callback(message);
}
