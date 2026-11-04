//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Protocol/MCP/Transport.h"
#include "llvm/ADT/StringRef.h"
#include <utility>

using namespace lldb_protocol::mcp;
using namespace llvm;

Transport::Transport(lldb_private::MainLoop &loop, lldb::IOObjectSP in,
                     lldb::IOObjectSP out, LogCallback log_callback)
    : JSONRPCTransport(loop, in, out), m_log_callback(std::move(log_callback)) {
}

void Transport::Log(StringRef message) {
  if (m_log_callback)
    m_log_callback(message);
}

llvm::Error Transport::ReplyWithParseError(StringRef raw_message,
                                           StringRef reason) {
  llvm::Expected<json::Value> value = json::parse(raw_message);
  if (!value) {
    // JSON-RPC forbids guessing an id, and malformed JSON carries none.
    consumeError(value.takeError());
    return llvm::Error::success();
  }

  const json::Object *object = value->getAsObject();
  if (!object)
    return llvm::Error::success();

  // A message without an id is a notification, which takes no response.
  const json::Value *raw_id = object->get("id");
  if (!raw_id)
    return llvm::Error::success();

  Id id;
  if (std::optional<StringRef> str = raw_id->getAsString())
    id = str->str();
  else if (std::optional<int64_t> num = raw_id->getAsInteger())
    id = *num;
  else
    return llvm::Error::success();

  return Send(Response{id, mcp::Error{eErrorCodeInvalidRequest, reason.str()}});
}
