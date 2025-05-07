//===-- DAPError.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAPError.h"
#include "llvm/Support/Error.h"
#include <system_error>

namespace lldb_dap {

char DAPError::ID;

DAPError::DAPError(std::string message, std::error_code EC, bool show_user,
                   std::optional<std::string> url,
                   std::optional<std::string> url_label)
    : m_message(message), m_ec(EC), m_show_user(show_user), m_url(url),
      m_url_label(url_label) {}

void DAPError::log(llvm::raw_ostream &OS) const { OS << m_message; }

std::error_code DAPError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

} // namespace lldb_dap
