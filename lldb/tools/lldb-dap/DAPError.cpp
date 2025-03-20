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

DAPError::DAPError(std::string message, bool show_user)
    : m_message(message), m_show_user(show_user) {}

void DAPError::log(llvm::raw_ostream &OS) const { OS << m_message; }

std::error_code DAPError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

} // namespace lldb_dap
