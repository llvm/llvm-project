//===-- DisconnectRequestHandler.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Protocol/ProtocolRequests.h"
#include "RequestHandler.h"
#include "llvm/Support/Error.h"
#include <optional>

using namespace llvm;
using namespace lldb_dap::protocol;

namespace lldb_dap {

/// Disconnect request; value of command field is 'disconnect'.
Error DisconnectRequestHandler::Run(
    const std::optional<DisconnectArguments> &arguments) const {
  bool terminateDebuggee = !dap.is_attach;

  if (arguments && arguments->terminateDebuggee)
    terminateDebuggee = *arguments->terminateDebuggee;

  if (Error error = dap.Disconnect(terminateDebuggee))
    return error;

  return Error::success();
}
} // namespace lldb_dap
