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
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <vector>

namespace lldb_mcp {

/// The multiplexer is the single MCP server a client talks to. It presents a
/// unified tool and resource surface and fans out to one or more backend LLDB
/// MCP servers, each reached through an mcp::Client.
///
/// Requests that can be answered locally (initialize, tools/list) are handled
/// directly. Requests that target a backend (tools/call, resources/list,
/// resources/read) are forwarded asynchronously, and the backend's answer is
/// relayed back to the client.
class Multiplexer {
public:
  template <typename T> using Reply = lldb_private::transport::Reply<T>;

  Multiplexer(
      std::unique_ptr<lldb_protocol::mcp::MCPTransport> client_transport,
      lldb_protocol::mcp::LogCallback log_callback = {});

  /// Adds a backend. The client must already be running (see Client::Run).
  /// Takes ownership.
  void AddBackend(std::unique_ptr<lldb_protocol::mcp::Client> backend);

  /// Registers the client-facing handlers. Backends should be added first.
  llvm::Error Run();

  /// Sets a handler invoked when the client disconnects (EOF).
  void SetDisconnectHandler(llvm::unique_function<void()> handler);

private:
  /// Locally-answered requests.
  /// @{
  llvm::Expected<lldb_protocol::mcp::InitializeResult>
  HandleInitialize(const lldb_protocol::mcp::InitializeParams &params);
  llvm::Expected<lldb_protocol::mcp::ListToolsResult> HandleToolsList();
  /// @}

  /// Requests forwarded to a backend.
  /// @{
  void HandleToolsCall(const lldb_protocol::mcp::CallToolParams &params,
                       Reply<lldb_protocol::mcp::CallToolResult> reply);
  void
  HandleResourcesList(Reply<lldb_protocol::mcp::ListResourcesResult> reply);
  void HandleResourcesRead(const lldb_protocol::mcp::ReadResourceParams &params,
                           Reply<lldb_protocol::mcp::ReadResourceResult> reply);
  /// @}

  /// Trampoline forwarding to the move-only disconnect handler.
  void HandleDisconnect();

  void Log(llvm::StringRef message);

  /// Returns the backend a request routes to, or null if none is available.
  /// All requests currently route to the single backend.
  lldb_protocol::mcp::Client *Route();

  std::unique_ptr<lldb_protocol::mcp::MCPTransport> m_client_transport;
  lldb_protocol::mcp::MCPBinderUP m_client_binder;
  lldb_protocol::mcp::LogCallback m_log_callback;

  std::vector<std::unique_ptr<lldb_protocol::mcp::Client>> m_backends;

  llvm::unique_function<void()> m_disconnect_handler;
};

} // namespace lldb_mcp

#endif
