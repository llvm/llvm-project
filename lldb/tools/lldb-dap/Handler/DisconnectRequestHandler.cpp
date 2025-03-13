//===-- DisconnectRequestHandler.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Protocol.h"
#include "RequestHandler.h"
#include "lldb/API/SBError.h"
#include "llvm/Support/Error.h"
#include <optional>
#include <system_error>
#include <variant>

llvm::Error takeError(lldb::SBError error) {
  if (error.Success())
    return llvm::Error::success();

  return llvm::createStringError(
      std::error_code(error.GetError(), std::generic_category()),
      error.GetCString());
}

namespace lldb_dap {

/// Disconnect request; value of command field is 'disconnect'.
llvm::Expected<protocol::DisconnectResponse> DisconnectRequestHandler::Run(
    const std::optional<protocol::DisconnectArguments> &arguments) const {
  bool terminateDebuggee = dap.is_attach ? false : true;

  if (arguments && arguments->terminateDebuggee)
    terminateDebuggee = *arguments->terminateDebuggee;

  lldb::SBError error = dap.Disconnect(terminateDebuggee);
  if (llvm::Error wrappedError = takeError(error))
    return wrappedError;
  return std::monostate{};
}
} // namespace lldb_dap
