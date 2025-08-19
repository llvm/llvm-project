//===-- JSONTransport.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/JSONTransport.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

char TransportUnhandledContentsError::ID;

TransportUnhandledContentsError::TransportUnhandledContentsError(
    std::string unhandled_contents)
    : m_unhandled_contents(unhandled_contents) {}

void TransportUnhandledContentsError::log(llvm::raw_ostream &OS) const {
  OS << "transport EOF with unhandled contents: '" << m_unhandled_contents
     << "'";
}
std::error_code TransportUnhandledContentsError::convertToErrorCode() const {
  return std::make_error_code(std::errc::bad_message);
}
