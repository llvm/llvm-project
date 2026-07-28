//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Protocol/MCP/Client.h"
#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Transport.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;
using namespace lldb_protocol::mcp;

Client::Client(std::unique_ptr<MCPTransport> transport,
               LogCallback log_callback)
    : m_transport(std::move(transport)),
      m_binder(std::make_unique<MCPBinder>(*m_transport)),
      m_log_callback(std::move(log_callback)) {
  m_initialize =
      m_binder->Bind<InitializeResult, InitializeParams>("initialize");
  m_tools_list = m_binder->Bind<ListToolsResult, void>("tools/list");
  m_tools_call = m_binder->Bind<CallToolResult, CallToolParams>("tools/call");
  m_resources_list =
      m_binder->Bind<ListResourcesResult, void>("resources/list");
  m_resources_read =
      m_binder->Bind<ReadResourceResult, ReadResourceParams>("resources/read");
  m_notify_initialized = m_binder->Bind<void>("notifications/initialized");
}

llvm::Error Client::Run() {
  m_binder->OnDisconnect(&Client::HandleDisconnect, this);
  m_binder->OnError(&Client::HandleError, this);
  return m_transport->RegisterMessageHandler(*m_binder);
}

void Client::SetDisconnectHandler(llvm::unique_function<void()> handler) {
  m_disconnect_handler = std::move(handler);
}

void Client::SetErrorHandler(llvm::unique_function<void(llvm::Error)> handler) {
  m_error_handler = std::move(handler);
}

void Client::HandleDisconnect() {
  if (m_disconnect_handler)
    m_disconnect_handler();
}

void Client::HandleError(llvm::Error error) {
  if (m_error_handler)
    m_error_handler(std::move(error));
  else
    consumeError(std::move(error));
}

void Client::Initialize(const InitializeParams &params,
                        Reply<InitializeResult> reply) {
  m_initialize(params, std::move(reply));
}

void Client::ToolsList(Reply<ListToolsResult> reply) {
  m_tools_list(std::move(reply));
}

void Client::ToolsCall(const CallToolParams &params,
                       Reply<CallToolResult> reply) {
  m_tools_call(params, std::move(reply));
}

void Client::ResourcesList(Reply<ListResourcesResult> reply) {
  m_resources_list(std::move(reply));
}

void Client::ResourcesRead(const ReadResourceParams &params,
                           Reply<ReadResourceResult> reply) {
  m_resources_read(params, std::move(reply));
}

void Client::NotifyInitialized() { m_notify_initialized(); }

void Client::CancelPendingRequests(llvm::StringRef reason) {
  m_binder->FailPendingRequests(reason);
}

void Client::Log(llvm::StringRef message) {
  if (m_log_callback)
    m_log_callback(message);
}
