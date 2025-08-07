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
#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Server.h"
#include <thread>

namespace lldb_private::mcp {

class ProtocolServerMCP : public ProtocolServer,
                          public lldb_protocol::mcp::Server {
public:
  ProtocolServerMCP();
  virtual ~ProtocolServerMCP() override;

  virtual llvm::Error Start(ProtocolServer::Connection connection) override;
  virtual llvm::Error Stop() override;

  static void Initialize();
  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "MCP"; }
  static llvm::StringRef GetPluginDescriptionStatic();

  static lldb::ProtocolServerUP CreateInstance();

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  Socket *GetSocket() const override { return m_listener.get(); }

private:
  void AcceptCallback(std::unique_ptr<Socket> socket);

  lldb_protocol::mcp::Capabilities GetCapabilities() override;

  bool m_running = false;

  MainLoop m_loop;
  std::thread m_loop_thread;

  std::unique_ptr<Socket> m_listener;
  std::vector<MainLoopBase::ReadHandleUP> m_listen_handlers;

  struct Client {
    lldb::IOObjectSP io_sp;
    MainLoopBase::ReadHandleUP read_handle_up;
    std::string buffer;
  };
  llvm::Error ReadCallback(Client &client);
  std::vector<std::unique_ptr<Client>> m_clients;
};
} // namespace lldb_private::mcp

#endif
