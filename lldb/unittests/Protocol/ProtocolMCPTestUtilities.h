//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_PROTOCOL_PROTOCOLMCPTESTUTILITIES_H
#define LLDB_UNITTESTS_PROTOCOL_PROTOCOLMCPTESTUTILITIES_H

#include "lldb/Protocol/MCP/Protocol.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h" // IWYU pragma: keep
#include "gtest/gtest.h"       // IWYU pragma: keep
#include <ostream>
#include <variant>

namespace lldb_protocol::mcp {

inline void PrintTo(const Request &req, std::ostream *os) {
  *os << llvm::formatv("{0}", toJSON(req)).str();
}

inline void PrintTo(const Response &resp, std::ostream *os) {
  *os << llvm::formatv("{0}", toJSON(resp)).str();
}

inline void PrintTo(const Notification &note, std::ostream *os) {
  *os << llvm::formatv("{0}", toJSON(note)).str();
}

inline void PrintTo(const Message &message, std::ostream *os) {
  return std::visit([os](auto &&message) { return PrintTo(message, os); },
                    message);
}

} // namespace lldb_protocol::mcp

#endif
