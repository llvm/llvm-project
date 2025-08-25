//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Protocol/MCP/Binder.h"
#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Transport.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include <atomic>
#include <cassert>
#include <mutex>

using namespace llvm;

namespace lldb_protocol::mcp {

/// Function object to reply to a call.
/// Each instance must be called exactly once, otherwise:
///  - the bug is logged, and (in debug mode) an assert will fire
///  - if there was no reply, an error reply is sent
///  - if there were multiple replies, only the first is sent
class ReplyOnce {
  std::atomic<bool> replied = {false};
  const Id id;
  MCPTransport *transport;               // Null when moved-from.
  MCPTransport::MessageHandler *handler; // Null when moved-from.

public:
  ReplyOnce(const Id id, MCPTransport *transport,
            MCPTransport::MessageHandler *handler)
      : id(id), transport(transport), handler(handler) {
    assert(handler);
  }
  ReplyOnce(ReplyOnce &&other)
      : replied(other.replied.load()), id(other.id), transport(other.transport),
        handler(other.handler) {
    other.transport = nullptr;
    other.handler = nullptr;
  }
  ReplyOnce &operator=(ReplyOnce &&) = delete;
  ReplyOnce(const ReplyOnce &) = delete;
  ReplyOnce &operator=(const ReplyOnce &) = delete;

  ~ReplyOnce() {
    if (transport && handler && !replied) {
      assert(false && "must reply to all calls!");
      (*this)(Response{id, Error{MCPError::kInternalError, "failed to reply",
                                 std::nullopt}});
    }
  }

  void operator()(const Response &resp) {
    assert(transport && handler && "moved-from!");
    if (replied.exchange(true)) {
      assert(false && "must reply to each call only once!");
      return;
    }

    if (llvm::Error error = transport->Send(Response{id, resp.result}))
      handler->OnError(std::move(error));
  }
};

Binder::RawHandler::RawHandler(MCPTransport *transport)
    : m_transport(transport) {}

void Binder::RawHandler::Received(const Notification &note) {
  std::scoped_lock<std::recursive_mutex> guard(m_mutex);
  auto it = m_notification_handlers.find(note.method);
  if (it == m_notification_handlers.end()) {
    OnError(llvm::createStringError(
        formatv("no handled for notification {0}", toJSON(note))));
    return;
  }
  it->second(note);
}

void Binder::RawHandler::Received(const Request &req) {
  ReplyOnce reply(req.id, m_transport, this);

  std::scoped_lock<std::recursive_mutex> guard(m_mutex);
  auto it = m_request_handlers.find(req.method);
  if (it == m_request_handlers.end()) {
    reply({req.id,
           Error{eErrorCodeMethodNotFound, "method not found", std::nullopt}});
    return;
  }

  it->second(req, std::move(reply));
}

void Binder::RawHandler::Received(const Response &resp) {
  std::scoped_lock<std::recursive_mutex> guard(m_mutex);
  auto it = m_pending_responses.find(resp.id);
  if (it == m_pending_responses.end()) {
    OnError(llvm::createStringError(
        formatv("no pending request for {0}", toJSON(resp))));
    return;
  }

  it->second(resp);
  m_pending_responses.erase(it);
}

void Binder::RawHandler::OnError(llvm::Error err) {
  std::scoped_lock<std::recursive_mutex> guard(m_mutex);
  if (m_error_handler)
    m_error_handler(m_transport, std::move(err));
}

void Binder::RawHandler::OnClosed() {
  std::scoped_lock<std::recursive_mutex> guard(m_mutex);
  if (m_disconnect_handler)
    m_disconnect_handler(m_transport);
}

void Binder::RawHandler::Send(
    const Request &req,
    llvm::unique_function<void(const Response &)> response_handler) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  Id id = ++m_seq;
  if (llvm::Error err = m_transport->Send(Request{id, req.method, req.params}))
    return OnError(std::move(err));
  m_pending_responses[id] = std::move(response_handler);
}

void Binder::RawHandler::Send(const Notification &note) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (llvm::Error err = m_transport->Send(note))
    return OnError(std::move(err));
}

} // namespace lldb_protocol::mcp
