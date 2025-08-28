//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Protocol/MCP/Transport.h"
#include "lldb/Host/JSONTransport.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include <string>
#include <utility>

using namespace llvm;
using namespace lldb;

namespace lldb_protocol::mcp {

Transport::Transport(IOObjectSP in, IOObjectSP out, std::string client_name,
                     LogCallback log_callback)
    : JSONRPCTransport(in, out), m_client_name(std::move(client_name)),
      m_log_callback(log_callback) {}

void Transport::Log(StringRef message) {
  if (m_log_callback)
    m_log_callback(formatv("{0}: {1}", m_client_name, message).str());
}

} // namespace lldb_protocol::mcp
