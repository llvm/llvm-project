//===-- SetInstructionBreakpointsRequestHandler.cpp -----------------------===//
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

/// Replaces all existing instruction breakpoints. Typically, instruction
/// breakpoints would be set from a disassembly window. To clear all instruction
/// breakpoints, specify an empty array. When an instruction breakpoint is hit,
/// a stopped event (with reason instruction breakpoint) is generated. Clients
/// should only call this request if the corresponding capability
/// supportsInstructionBreakpoints is true.
llvm::Expected<protocol::SetInstructionBreakpointsResponseBody>
SetInstructionBreakpointsRequestHandler::Run(
    const protocol::SetInstructionBreakpointsArguments &args) const {
  std::vector<protocol::Breakpoint> response_breakpoints;

  // Disable any instruction breakpoints that aren't in this request.
  // There is no call to remove instruction breakpoints other than calling this
  // function with a smaller or empty "breakpoints" list.
  llvm::DenseSet<lldb::addr_t> seen(
      llvm::from_range, llvm::make_first_range(dap.instruction_breakpoints));

  for (const auto &bp : args.breakpoints) {
    // Read instruction breakpoint request.
    InstructionBreakpoint inst_bp(dap, bp);
    const auto [iv, inserted] = dap.instruction_breakpoints.try_emplace(
        inst_bp.GetInstructionAddressReference(), dap, bp);
    if (inserted)
      iv->second.SetBreakpoint();
    else
      iv->second.UpdateBreakpoint(inst_bp);
    response_breakpoints.push_back(iv->second.ToProtocolBreakpoint());
    seen.erase(inst_bp.GetInstructionAddressReference());
  }

  for (const auto &addr : seen) {
    auto inst_bp = dap.instruction_breakpoints.find(addr);
    if (inst_bp == dap.instruction_breakpoints.end())
      continue;
    dap.target.BreakpointDelete(inst_bp->second.GetID());
    dap.instruction_breakpoints.erase(addr);
  }

  return protocol::SetInstructionBreakpointsResponseBody{
      std::move(response_breakpoints)};
}

} // namespace lldb_dap
