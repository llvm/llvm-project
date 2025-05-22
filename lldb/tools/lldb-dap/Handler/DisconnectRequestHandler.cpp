//===-- DisconnectRequestHandler.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "LLDBUtils.h"
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
  bool terminate_debuggee = dap.is_attach ? false : true;
  bool keep_suspend = false;

  if (arguments) {
    terminate_debuggee = arguments->terminateDebuggee;
    keep_suspend = !arguments->suspendDebuggee;
  }

  if (Error error = dap.Disconnect(terminate_debuggee, keep_suspend))
    return error;

  return Error::success();
}
} // namespace lldb_dap
