//===-- SetFunctionBreakpointsRequestHandler.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "RequestHandler.h"

namespace lldb_dap {

/// Replaces all existing function breakpoints with new function breakpoints.
/// To clear all function breakpoints, specify an empty array.
/// When a function breakpoint is hit, a stopped event (with reason function
/// breakpoint) is generated. Clients should only call this request if the
/// corresponding capability supportsFunctionBreakpoints is true.
llvm::Expected<protocol::SetFunctionBreakpointsResponseBody>
SetFunctionBreakpointsRequestHandler::Run(
    const protocol::SetFunctionBreakpointsArguments &args) const {
  std::vector<protocol::Breakpoint> response_breakpoints;

  // Disable any function breakpoints that aren't in this request.
  // There is no call to remove function breakpoints other than calling this
  // function with a smaller or empty "breakpoints" list.
  const auto name_iter = dap.function_breakpoints.keys();
  llvm::DenseSet<llvm::StringRef> seen(name_iter.begin(), name_iter.end());
  for (const auto &fb : args.breakpoints) {
    FunctionBreakpoint fn_bp(dap, fb);
    const auto [it, inserted] =
        dap.function_breakpoints.try_emplace(fn_bp.GetFunctionName(), dap, fb);
    if (inserted)
      it->second.SetBreakpoint();
    else
      it->second.UpdateBreakpoint(fn_bp);

    response_breakpoints.push_back(it->second.ToProtocolBreakpoint());
    seen.erase(fn_bp.GetFunctionName());
  }

  // Remove any breakpoints that are no longer in our list
  for (const auto &name : seen) {
    auto fn_bp = dap.function_breakpoints.find(name);
    if (fn_bp == dap.function_breakpoints.end())
      continue;
    dap.target.BreakpointDelete(fn_bp->second.GetID());
    dap.function_breakpoints.erase(name);
  }

  return protocol::SetFunctionBreakpointsResponseBody{
      std::move(response_breakpoints)};
}

} // namespace lldb_dap
