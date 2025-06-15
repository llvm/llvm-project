//===- ProtocolServerMCP.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_PROTOCOL_MCP_PROTOCOLSERVERMCP_H
#define LLDB_PLUGINS_PROTOCOL_MCP_PROTOCOLSERVERMCP_H

#include "Protocol.h"
#include "Tool.h"
#include "lldb/Core/ProtocolServer.h"
#include "lldb/Host/JSONTransport.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/Socket.h"
#include "llvm/ADT/StringMap.h"
#include <thread>

namespace lldb_private::mcp {

class ProtocolServerMCP : public ProtocolServer {
public:
  ProtocolServerMCP(Debugger &debugger);
  virtual ~ProtocolServerMCP() override;

  virtual llvm::Error Start(ProtocolServer::Connection connection) override;
  virtual llvm::Error Stop() override;

  static void Initialize();
  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "MCP"; }
  static llvm::StringRef GetPluginDescriptionStatic();

  static lldb::ProtocolServerSP CreateInstance(Debugger &debugger);

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  Socket *GetSocket() const override { return m_listener.get(); }

protected:
  void AddTool(std::unique_ptr<Tool> tool);
  void AddHandler(
      llvm::StringRef method,
      std::function<protocol::Response(const protocol::Request &)> handler);

private:
  llvm::Error Run(std::unique_ptr<JSONTransport> transport);
  llvm::Expected<protocol::Response> Handle(protocol::Request request);

  protocol::Response InitializeHandler(const protocol::Request &);
  protocol::Response ToolsListHandler(const protocol::Request &);
  protocol::Response ToolsCallHandler(const protocol::Request &);

  protocol::Capabilities GetCapabilities();

  llvm::StringLiteral kName = "lldb-mcp";
  llvm::StringLiteral kVersion = "0.1.0";

  Debugger &m_debugger;

  bool m_running = false;

  MainLoop m_loop;
  std::thread m_loop_thread;

  std::unique_ptr<Socket> m_listener;
  std::vector<MainLoopBase::ReadHandleUP> m_read_handles;
  std::vector<std::pair<lldb::IOObjectSP, std::thread>> m_clients;

  std::mutex m_server_mutex;
  llvm::StringMap<std::unique_ptr<Tool>> m_tools;
  llvm::StringMap<std::function<protocol::Response(const protocol::Request &)>>
      m_handlers;
};
} // namespace lldb_private::mcp

#endif
