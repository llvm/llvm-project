//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAPError.h"
#include "Protocol/ProtocolRequests.h"
#include "RequestHandler.h"
#include "llvm/Support/Error.h"

using namespace lldb_dap;
using namespace lldb_dap::protocol;

llvm::Error UnknownRequestHandler::Run(const UnknownArguments &args) const {
  return llvm::make_error<DAPError>("unknown request");
}
