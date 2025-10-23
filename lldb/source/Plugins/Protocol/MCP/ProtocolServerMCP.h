//===- ProtocolServerMCP.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_PROTOCOL_MCP_PROTOCOLSERVERMCP_H
#define LLDB_PLUGINS_PROTOCOL_MCP_PROTOCOLSERVERMCP_H

#include "lldb/Core/ProtocolServer.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/Socket.h"
#include "lldb/Protocol/MCP/Server.h"
#include "lldb/Protocol/MCP/Transport.h"
#include <map>
#include <memory>
#include <thread>
#include <tuple>
#include <vector>

namespace lldb_private::mcp {

class ProtocolServerMCP : public ProtocolServer {

  using ServerUP = std::unique_ptr<lldb_protocol::mcp::Server>;

  using ReadHandleUP = MainLoop::ReadHandleUP;

public:
  ProtocolServerMCP();
  ~ProtocolServerMCP() override;

  llvm::Error Start(ProtocolServer::Connection connection) override;
  llvm::Error Stop() override;

  static void Initialize();
  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "MCP"; }
  static llvm::StringRef GetPluginDescriptionStatic();

  static lldb::ProtocolServerUP CreateInstance();

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  Socket *GetSocket() const override { return m_listener.get(); }

protected:
  // This adds tools and resource providers that
  // are specific to this server. Overridable by the unit tests.
  virtual void Extend(lldb_protocol::mcp::Server &server) const;

private:
  void AcceptCallback(std::unique_ptr<Socket> socket);

  bool m_running = false;

  lldb_private::MainLoop m_loop;
  std::thread m_loop_thread;
  std::mutex m_mutex;
  size_t m_client_count = 0;

  std::unique_ptr<Socket> m_listener;
  std::vector<ReadHandleUP> m_accept_handles;

  ServerUP m_server;
  lldb_protocol::mcp::ServerInfoHandle m_server_info_handle;
};

} // namespace lldb_private::mcp

#endif
