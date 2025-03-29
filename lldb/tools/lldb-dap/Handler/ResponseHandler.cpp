//===-- ResponseHandler.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ResponseHandler.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace lldb_dap {

void UnknownResponseHandler::operator()(
    llvm::Expected<llvm::json::Value> value) const {
  llvm::errs() << "unexpected response: ";
  if (value) {
    if (std::optional<llvm::StringRef> str = value->getAsString())
      llvm::errs() << *str;
  } else {
    llvm::errs() << "error: " << llvm::toString(value.takeError());
  }
  llvm::errs() << '\n';
}

void LogFailureResponseHandler::operator()(
    llvm::Expected<llvm::json::Value> value) const {
  if (!value)
    llvm::errs() << "reverse request \"" << m_command << "\" (" << m_id
                 << ") failed: " << llvm::toString(value.takeError()) << '\n';
}

} // namespace lldb_dap
