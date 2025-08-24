//===-- SetBreakpointsRequestHandler.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Protocol/ProtocolRequests.h"
#include "RequestHandler.h"

namespace lldb_dap {

/// Sets multiple breakpoints for a single source and clears all previous
/// breakpoints in that source. To clear all breakpoint for a source, specify an
/// empty array. When a breakpoint is hit, a `stopped` event (with reason
/// `breakpoint`) is generated.
llvm::Expected<protocol::SetBreakpointsResponseBody>
SetBreakpointsRequestHandler::Run(
    const protocol::SetBreakpointsArguments &args) const {
  std::vector<protocol::Breakpoint> response_breakpoints =
      dap.SetSourceBreakpoints(args.source, args.breakpoints);
  return protocol::SetBreakpointsResponseBody{std::move(response_breakpoints)};
}

} // namespace lldb_dap
