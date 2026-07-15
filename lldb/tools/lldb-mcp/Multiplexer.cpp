//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Multiplexer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include <map>
#include <memory>

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

/// Backend-local URI prefixes.
/// @{
constexpr llvm::StringLiteral kDebuggerLocalPrefix = "lldb-mcp://debugger/";
constexpr llvm::StringLiteral kResourceLocalPrefix = "lldb://debugger/";
/// @}

std::string replaceAll(StringRef text, StringRef from, StringRef to) {
  std::string result;
  size_t pos = 0;
  while (true) {
    size_t next = text.find(from, pos);
    if (next == StringRef::npos) {
      result += text.substr(pos).str();
      break;
    }
    result += text.substr(pos, next - pos).str();
    result += to.str();
    pos = next + from.size();
  }
  return result;
}

CallToolResult makeTextResult(std::string text) {
  CallToolResult result;
  result.content.emplace_back(TextContent{{std::move(text)}});
  return result;
}

} // namespace

std::string lldb_mcp::RewriteURIsToGlobal(StringRef text, lldb::pid_t pid) {
  std::string debugger_global =
      formatv("lldb-mcp://instance/{0}/debugger/", pid).str();
  std::string resource_global =
      formatv("lldb://instance/{0}/debugger/", pid).str();
  std::string step = replaceAll(text, kDebuggerLocalPrefix, debugger_global);
  return replaceAll(step, kResourceLocalPrefix, resource_global);
}

std::optional<RoutedURI> lldb_mcp::ParseGlobalURI(StringRef uri) {
  size_t scheme_end = uri.find("://");
  if (scheme_end == StringRef::npos)
    return std::nullopt;

  StringRef scheme = uri.take_front(scheme_end);
  StringRef rest = uri.drop_front(scheme_end + 3);
  if (!rest.consume_front("instance/"))
    return std::nullopt;

  size_t slash = rest.find('/');
  StringRef pid_str = slash == StringRef::npos ? rest : rest.take_front(slash);
  lldb::pid_t pid;
  if (pid_str.getAsInteger(10, pid))
    return std::nullopt;

  StringRef tail =
      slash == StringRef::npos ? StringRef() : rest.drop_front(slash + 1);
  RoutedURI routed;
  routed.pid = pid;
  routed.local = formatv("{0}://{1}", scheme, tail).str();
  return routed;
}

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

void Multiplexer::AddBackend(lldb::pid_t pid, std::unique_ptr<Client> backend) {
  // A backend that disconnects or errors mid-request will never answer its
  // forwarded requests. Fail them so the client sees an error instead of
  // waiting forever, since there is no request timeout.
  Client *client = backend.get();
  client->SetDisconnectHandler(
      [client] { client->CancelPendingRequests("backend disconnected"); });
  client->SetErrorHandler([client](llvm::Error error) {
    consumeError(std::move(error));
    client->CancelPendingRequests("backend transport error");
  });
  m_backends.push_back(Backend{pid, std::move(backend)});
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

void Multiplexer::Shutdown() {
  for (Backend &backend : m_backends)
    backend.client->CancelPendingRequests("lldb-mcp is shutting down");
}

Client *Multiplexer::RouteToPid(lldb::pid_t pid) {
  for (Backend &backend : m_backends)
    if (backend.pid == pid)
      return backend.client.get();
  return nullptr;
}

Client *Multiplexer::First() {
  return m_backends.empty() ? nullptr : m_backends.front().client.get();
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
            json::Object{
                {"type", "string"},
                {"description",
                 "The debugger URI selecting the debug session, as reported by "
                 "sessions_list, e.g. lldb-mcp://instance/{pid}/debugger/{id}. "
                 "Defaults to the first session."}}},
       }},
      {"required", json::Array{"command"}},
  };
  result.tools.push_back(std::move(command));

  ToolDefinition sessions_list;
  sessions_list.name = kToolSessionsList;
  sessions_list.description =
      "List the active debug sessions across all lldb instances.";
  sessions_list.inputSchema = json::Object{{"type", "object"}};
  result.tools.push_back(std::move(sessions_list));

  return result;
}

void Multiplexer::HandleToolsCall(const CallToolParams &params,
                                  Reply<CallToolResult> reply) {
  if (params.name == kToolCommand)
    return HandleCommand(params, std::move(reply));
  if (params.name == kToolSessionsList)
    return HandleSessionsList(std::move(reply));
  reply(createStringError(formatv("no tool \"{0}\"", params.name)));
}

void Multiplexer::HandleCommand(const CallToolParams &params,
                                Reply<CallToolResult> reply) {
  json::Object args;
  if (params.arguments)
    if (const json::Object *object = params.arguments->getAsObject())
      args = *object;

  std::string debugger_arg;
  if (std::optional<StringRef> debugger = args.getString("debugger"))
    debugger_arg = debugger->str();

  Client *backend = nullptr;
  if (debugger_arg.empty()) {
    // Default to the first session, letting the backend pick its debugger.
    backend = First();
    args.erase("debugger");
  } else {
    std::optional<RoutedURI> routed = ParseGlobalURI(debugger_arg);
    if (!routed)
      return reply(createStringError(
          formatv("malformed debugger uri \"{0}\"", debugger_arg)));
    backend = RouteToPid(routed->pid);
    args["debugger"] = routed->local;
  }

  if (!backend)
    return reply(createStringError("no debug session available"));

  CallToolParams backend_params;
  backend_params.name = kBackendToolCommand;
  backend_params.arguments = json::Value(std::move(args));
  backend->ToolsCall(backend_params, std::move(reply));
}

void Multiplexer::HandleSessionsList(Reply<CallToolResult> reply) {
  if (m_backends.empty())
    return reply(makeTextResult(""));

  // All backends share the multiplexer's MainLoop, so these reply callbacks run
  // serially on one thread and the aggregation state below needs no locking.
  struct State {
    size_t remaining;
    // Keyed by pid so the aggregated output is deterministic.
    std::map<lldb::pid_t, std::string> texts;
    std::string error;
    Reply<CallToolResult> reply;
  };
  auto state = std::make_shared<State>();
  state->remaining = m_backends.size();
  state->reply = std::move(reply);

  for (Backend &backend : m_backends) {
    lldb::pid_t pid = backend.pid;
    CallToolParams params;
    params.name = kBackendToolDebuggerList;
    backend.client->ToolsCall(
        params, [state, pid](Expected<CallToolResult> result) {
          if (result) {
            std::string text;
            for (const TextContent &content : result->content)
              text += RewriteURIsToGlobal(content.text, pid);
            state->texts[pid] = std::move(text);
          } else if (state->error.empty()) {
            state->error = toString(result.takeError());
          } else {
            consumeError(result.takeError());
          }

          if (--state->remaining != 0)
            return;
          if (!state->error.empty())
            return state->reply(createStringError(state->error));
          std::string combined;
          for (const auto &entry : state->texts)
            combined += entry.second;
          state->reply(makeTextResult(combined));
        });
  }
}

void Multiplexer::HandleResourcesList(Reply<ListResourcesResult> reply) {
  if (m_backends.empty())
    return reply(ListResourcesResult{});

  // These reply callbacks run serially on the shared MainLoop thread, so this
  // state needs no locking.
  struct State {
    size_t remaining;
    std::map<lldb::pid_t, std::vector<Resource>> resources;
    std::string error;
    Reply<ListResourcesResult> reply;
  };
  auto state = std::make_shared<State>();
  state->remaining = m_backends.size();
  state->reply = std::move(reply);

  for (Backend &backend : m_backends) {
    lldb::pid_t pid = backend.pid;
    backend.client->ResourcesList(
        [state, pid](Expected<ListResourcesResult> result) {
          if (result) {
            std::vector<Resource> rewritten;
            for (Resource resource : result->resources) {
              resource.uri = RewriteURIsToGlobal(resource.uri, pid);
              rewritten.push_back(std::move(resource));
            }
            state->resources[pid] = std::move(rewritten);
          } else if (state->error.empty()) {
            state->error = toString(result.takeError());
          } else {
            consumeError(result.takeError());
          }

          if (--state->remaining != 0)
            return;
          if (!state->error.empty())
            return state->reply(createStringError(state->error));
          ListResourcesResult combined;
          for (auto &entry : state->resources)
            for (Resource &resource : entry.second)
              combined.resources.push_back(std::move(resource));
          state->reply(std::move(combined));
        });
  }
}

void Multiplexer::HandleResourcesRead(const ReadResourceParams &params,
                                      Reply<ReadResourceResult> reply) {
  std::optional<RoutedURI> routed = ParseGlobalURI(params.uri);
  if (!routed)
    return reply(createStringError(
        formatv("malformed resource uri \"{0}\"", params.uri)));

  Client *backend = RouteToPid(routed->pid);
  if (!backend)
    return reply(createStringError(formatv("no instance {0}", routed->pid)));

  ReadResourceParams backend_params;
  backend_params.uri = routed->local;
  lldb::pid_t pid = routed->pid;
  backend->ResourcesRead(
      backend_params, [reply = std::move(reply),
                       pid](Expected<ReadResourceResult> result) mutable {
        if (result)
          for (TextResourceContents &content : result->contents)
            content.uri = RewriteURIsToGlobal(content.uri, pid);
        reply(std::move(result));
      });
}

void Multiplexer::HandleDisconnect() {
  if (m_disconnect_handler)
    m_disconnect_handler();
}

void Multiplexer::Log(llvm::StringRef message) {
  if (m_log_callback)
    m_log_callback(message);
}
