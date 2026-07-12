//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_MCP_MULTIPLEXER_H
#define LLDB_TOOLS_LLDB_MCP_MULTIPLEXER_H

#include "lldb/Protocol/MCP/Client.h"
#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Transport.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lldb_mcp {

/// Rewrites the backend-local debugger and resource URIs found in \p text to
/// their instance-qualified global form for instance \p pid, e.g.
/// `lldb-mcp://debugger/1` becomes `lldb-mcp://instance/{pid}/debugger/1`.
std::string RewriteURIsToGlobal(llvm::StringRef text, lldb::pid_t pid);

/// An instance-qualified URI decomposed for routing.
struct RoutedURI {
  lldb::pid_t pid;
  /// The backend-local URI, with the `instance/{pid}` segment stripped.
  std::string local;
};

/// Parses an instance-qualified global URI (`<scheme>://instance/{pid}/...`)
/// into its instance pid and backend-local URI. Returns std::nullopt if \p uri
/// is not instance-qualified.
std::optional<RoutedURI> ParseGlobalURI(llvm::StringRef uri);

/// The multiplexer is the single MCP server a client talks to. It presents a
/// unified tool and resource surface and fans out to one or more backend LLDB
/// MCP servers, each reached through an mcp::Client and identified by the pid
/// of its lldb instance.
///
/// Requests answered locally (initialize, tools/list) are handled directly.
/// Requests that list across instances (sessions_list, resources/list) fan out
/// to every backend and aggregate. Requests that target one instance (command,
/// resources/read) are routed by an instance-qualified URI of the form
/// `lldb-mcp://instance/{pid}/debugger/{id}` (tools) or
/// `lldb://instance/{pid}/debugger/{id}/target/{idx}` (resources). Backends
/// only know their local `lldb-mcp://debugger/{id}` form, so URIs are rewritten
/// in both directions.
class Multiplexer {
public:
  template <typename T> using Reply = lldb_private::transport::Reply<T>;

  Multiplexer(
      std::unique_ptr<lldb_protocol::mcp::MCPTransport> client_transport,
      lldb_protocol::mcp::LogCallback log_callback = {});

  /// Adds a backend identified by the pid of its lldb instance. The client must
  /// already be running (see Client::Run). Takes ownership.
  void AddBackend(lldb::pid_t pid,
                  std::unique_ptr<lldb_protocol::mcp::Client> backend);

  /// Registers the client-facing handlers. Backends should be added first.
  llvm::Error Run();

  /// Fails any requests still in flight to a backend. Call after the client has
  /// disconnected, before teardown.
  void Shutdown();

  /// Sets a handler invoked when the client disconnects (EOF).
  void SetDisconnectHandler(llvm::unique_function<void()> handler);

private:
  struct Backend {
    lldb::pid_t pid;
    std::unique_ptr<lldb_protocol::mcp::Client> client;
  };

  /// Locally-answered requests.
  /// @{
  llvm::Expected<lldb_protocol::mcp::InitializeResult>
  HandleInitialize(const lldb_protocol::mcp::InitializeParams &params);
  llvm::Expected<lldb_protocol::mcp::ListToolsResult> HandleToolsList();
  /// @}

  /// Requests routed to or aggregated across backends.
  /// @{
  void HandleToolsCall(const lldb_protocol::mcp::CallToolParams &params,
                       Reply<lldb_protocol::mcp::CallToolResult> reply);
  void HandleCommand(const lldb_protocol::mcp::CallToolParams &params,
                     Reply<lldb_protocol::mcp::CallToolResult> reply);
  void HandleSessionsList(Reply<lldb_protocol::mcp::CallToolResult> reply);
  void
  HandleResourcesList(Reply<lldb_protocol::mcp::ListResourcesResult> reply);
  void HandleResourcesRead(const lldb_protocol::mcp::ReadResourceParams &params,
                           Reply<lldb_protocol::mcp::ReadResourceResult> reply);
  /// @}

  /// Trampoline forwarding to the move-only disconnect handler.
  void HandleDisconnect();

  void Log(llvm::StringRef message);

  /// Returns the backend for \p pid, or nullptr if there is none.
  lldb_protocol::mcp::Client *RouteToPid(lldb::pid_t pid);
  /// Returns the first backend, or nullptr if there are none.
  lldb_protocol::mcp::Client *First();

  std::unique_ptr<lldb_protocol::mcp::MCPTransport> m_client_transport;
  lldb_protocol::mcp::MCPBinderUP m_client_binder;
  lldb_protocol::mcp::LogCallback m_log_callback;

  std::vector<Backend> m_backends;

  llvm::unique_function<void()> m_disconnect_handler;
};

} // namespace lldb_mcp

#endif
