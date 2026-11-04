//===-- InitializeRequestHandler.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandPlugins.h"
#include "DAP.h"
#include "EventHelper.h"
#include "JSONUtils.h"
#include "LLDBUtils.h"
#include "Protocol/ProtocolRequests.h"
#include "RequestHandler.h"
#include "lldb/API/SBTarget.h"

using namespace lldb_dap;
using namespace lldb_dap::protocol;

/// Initialize request; value of command field is 'initialize'.
llvm::Expected<InitializeResponse> InitializeRequestHandler::Run(
    const InitializeRequestArguments &arguments) const {
  // Store initialization arguments for later use in Launch/Attach.
  dap.clientFeatures = arguments.supportedFeatures;
  dap.sourceInitFile = arguments.lldbExtSourceInitFile;

  return dap.GetCapabilities();
}
