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
#include "llvm/ADT/SmallVector.h"
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
/// Backends are of two kinds. **Remote** backends are external lldb processes
/// the user already runs, reached over sockets. The **local** backend is an
/// embedded MCP server in this process that hosts the debug sessions the
/// multiplexer creates. Managed sessions are its debuggers.
///
/// Requests answered locally (initialize, tools/list) are handled directly.
/// Listing requests (sessions_list, resources/list) fan out to every backend
/// and aggregate. Targeted requests (command, resources/read) are routed by the
/// pid parsed from an instance-qualified URI. Session management
/// (session_create/session_close) operates on the local backend. Backends only
/// know their local `lldb-mcp://debugger/{id}` form, so URIs are rewritten in
/// both directions.
class Multiplexer {
public:
  template <typename T> using Reply = lldb_private::transport::Reply<T>;

  Multiplexer(
      std::unique_ptr<lldb_protocol::mcp::MCPTransport> client_transport,
      lldb_protocol::mcp::LogCallback log_callback = {});

  /// Adds a remote backend (an external lldb instance) identified by its pid.
  /// The client must already be running (see Client::Run). Takes ownership.
  void AddBackend(lldb::pid_t pid,
                  std::unique_ptr<lldb_protocol::mcp::Client> backend);

  /// Adds the local backend: the in-process embedded server that hosts managed
  /// sessions. \p pid is this process's pid. Takes ownership.
  void AddLocalBackend(lldb::pid_t pid,
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
    /// True for the in-process backend that hosts managed sessions.
    bool local = false;
    /// Cleared when the backend disconnects. A dead backend is skipped by
    /// routing and aggregation rather than left to hang a pending request.
    bool alive = true;
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
  void HandleSessionCreate(Reply<lldb_protocol::mcp::CallToolResult> reply);
  void HandleSessionClose(const lldb_protocol::mcp::CallToolParams &params,
                          Reply<lldb_protocol::mcp::CallToolResult> reply);
  void
  HandleResourcesList(Reply<lldb_protocol::mcp::ListResourcesResult> reply);
  void HandleResourcesRead(const lldb_protocol::mcp::ReadResourceParams &params,
                           Reply<lldb_protocol::mcp::ReadResourceResult> reply);
  /// @}

  /// Trampoline forwarding to the move-only disconnect handler.
  void HandleDisconnect();

  void Log(llvm::StringRef message);

  /// Returns the backend for \p pid, or nullptr if there is none (or it is
  /// no longer alive).
  lldb_protocol::mcp::Client *RouteToPid(lldb::pid_t pid);
  /// Returns the local (in-process) backend, or nullptr if there is none.
  Backend *LocalBackend();
  /// Returns the backends that are still alive, in registration order.
  llvm::SmallVector<Backend *> LiveBackends();
  /// Installs the disconnect and error handlers that retire the backend for
  /// \p pid, so a dropped or erroring backend's in-flight requests are failed
  /// rather than left hanging (there is no request timeout).
  void InstallBackendHandlers(lldb::pid_t pid,
                              lldb_protocol::mcp::Client &backend);
  /// Marks the backend for \p pid dead and fails its in-flight requests, so a
  /// backend that disconnects mid-request doesn't strand a pending reply.
  void RetireBackend(lldb::pid_t pid);

  std::unique_ptr<lldb_protocol::mcp::MCPTransport> m_client_transport;
  lldb_protocol::mcp::MCPBinderUP m_client_binder;
  lldb_protocol::mcp::LogCallback m_log_callback;

  std::vector<Backend> m_backends;

  llvm::unique_function<void()> m_disconnect_handler;
};

} // namespace lldb_mcp

#endif
