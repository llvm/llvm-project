//===-- MCPError.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCPError.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

namespace lldb_private::mcp {

char MCPError::ID;

MCPError::MCPError(std::string message, int64_t error_code)
    : m_message(message), m_error_code(error_code) {}

void MCPError::log(llvm::raw_ostream &OS) const { OS << m_message; }

std::error_code MCPError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

protocol::Error MCPError::toProtcolError() const {
  return protocol::Error{m_error_code, m_message, std::nullopt};
}

} // namespace lldb_private::mcp
