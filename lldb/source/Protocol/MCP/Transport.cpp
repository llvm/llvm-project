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

Transport::Transport(lldb::IOObjectSP in, lldb::IOObjectSP out,
                     LogCallback log_callback)
    : JSONRPCTransport(in, out), m_log_callback(std::move(log_callback)) {}

void Transport::Log(StringRef message) {
  if (m_log_callback)
    m_log_callback(message);
}
